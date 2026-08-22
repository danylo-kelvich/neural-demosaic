import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfa import XTRANS_PATTERN, make_mosaic
from src.models import PackedXTransNet
from src.demosaic_opencl import MarkesteijnOpenCLDemosaicer
import functools
print = functools.partial(print, flush=True)

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
GT_DIR = Path("data/gt")
CKPT = Path("weights/packed.pt")
PATCH = 96
BATCH = 32
ITERS = 50000
LR = 1e-3
LOG_EVERY = 50
VAL_EVERY = 500


class GTPatches(Dataset):
    def __init__(self, files):
        self.images = [np.load(f, mmap_mode="r") for f in files]

    def __len__(self):
        return ITERS * BATCH

    def __getitem__(self, _):
        img = self.images[torch.randint(len(self.images), ()).item()]
        h, w, _ = img.shape
        y = torch.randint(h - PATCH + 1, ()).item()
        x = torch.randint(w - PATCH + 1, ()).item()
        patch = img[y:y + PATCH, x:x + PATCH].astype(np.float32)

        if torch.rand(()) < 0.5:
            patch = patch[::-1]
        if torch.rand(()) < 0.5:
            patch = patch[:, ::-1]
        if torch.rand(()) < 0.5:
            patch = patch.transpose(1, 0, 2)
        patch = np.ascontiguousarray(patch)

        mosaic = make_mosaic(patch, XTRANS_PATTERN)
        return torch.from_numpy(mosaic)[None], torch.from_numpy(patch).permute(2, 0, 1)


def gamma(x):
    return x.clamp_min(0).add(1e-3).pow(1 / 2.4)


def psnr(a, b):
    return -10 * torch.log10(F.mse_loss(a, b)).item()


def make_eval_pairs(images, size=768):
    pairs = []
    for img in images:
        h, w, _ = img.shape
        s = min(size, min(h, w) // 6 * 6)
        y, x = (h - s) // 2, (w - s) // 2
        crop = np.ascontiguousarray(img[y:y + s, x:x + s], dtype=np.float32)
        gt = torch.from_numpy(crop).permute(2, 0, 1)[None]
        mosaic = torch.from_numpy(make_mosaic(crop, XTRANS_PATTERN))[None, None]
        pairs.append((mosaic, gt))
    return pairs


@torch.no_grad()
def evaluate(fn, pairs):
    return float(np.mean([psnr(fn(mo.to(DEVICE)), gt.to(DEVICE)) for mo, gt in pairs]))


def markesteijn_psnr(kernels, pairs):
    pattern = XTRANS_PATTERN.astype(np.uint8)
    vals = []
    for mo, gt in pairs:
        mosaic = np.pad(mo[0, 0].numpy(), 18, mode="symmetric")
        out = kernels.demosaic(raw_image=mosaic, xtrans_pattern=pattern, passes=3, crop=False)
        out = torch.from_numpy(np.ascontiguousarray(out[18:-18, 18:-18])).permute(2, 0, 1)[None]
        vals.append(psnr(out, gt))
    return float(np.mean(vals))


def main():
    files = sorted(GT_DIR.glob("*.npy"))
    assert len(files) >= 2, f"need GT files in {GT_DIR}"
    n_val = max(1, len(files) // 8)
    train_files, val_files = files[:-n_val], files[-n_val:]
    print(f"{len(train_files)} train images, val: {[f.name for f in val_files]}")

    loader = DataLoader(GTPatches(train_files), batch_size=BATCH, num_workers=4,
                        persistent_workers=True)
    model = PackedXTransNet(XTRANS_PATTERN, width=32, depth=8).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ITERS)

    train_eval = make_eval_pairs([np.load(f, mmap_mode="r") for f in train_files])
    val_eval = make_eval_pairs([np.load(f, mmap_mode="r") for f in val_files])
    base_train = evaluate(model.baseline, train_eval)
    base_val = evaluate(model.baseline, val_eval)
    print(f"bilinear PSNR:    train {base_train:.2f}, val {base_val:.2f}")

    os.environ.setdefault("PYOPENCL_CTX", "0")
    kernels = MarkesteijnOpenCLDemosaicer(kernel_path="src/kernels/demosaic_markesteijn.cl")
    mark_train = markesteijn_psnr(kernels, train_eval)
    mark_val = markesteijn_psnr(kernels, val_eval)
    print(f"markesteijn PSNR: train {mark_train:.2f}, val {mark_val:.2f}")

    CKPT.parent.mkdir(exist_ok=True)
    best_psnr = 0.0
    running = 0.0
    t0 = time.time()

    for step, (mosaic, gt) in enumerate(loader, 1):
        mosaic, gt = mosaic.to(DEVICE), gt.to(DEVICE)
        out = model(mosaic)
        loss = F.l1_loss(out, gt) + F.l1_loss(gamma(out), gamma(gt))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        running += loss.item()

        if step % LOG_EVERY == 0:
            print(f"step {step}/{ITERS}  loss {running / LOG_EVERY:.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {LOG_EVERY * BATCH / (time.time() - t0):.0f} img/s")
            running = 0.0
            t0 = time.time()

        if step % VAL_EVERY == 0:
            model.eval()
            train_psnr = evaluate(model, train_eval)
            val_psnr = evaluate(model, val_eval)
            model.train()
            marker = ""
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save(model.state_dict(), CKPT)
                marker = " *"
            print(f"PSNR train {train_psnr:.2f} (bilin {base_train:.2f}, mark {mark_train:.2f}) | "
                  f"val {val_psnr:.2f} (bilin {base_val:.2f}, mark {mark_val:.2f}, best {best_psnr:.2f}){marker}")

        if step >= ITERS:
            break


if __name__ == "__main__":
    main()
