import sys
import time
from pathlib import Path

import numpy as np
import rawler_py
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.cfa import XTRANS_PATTERN, BAYER_PATTERN
from src.models import PackedXTransNet
from src.utils import apply_raw_processing

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
CKPT = "weights/packed_5183_3208.pt"
RAF_PATH = "samples/DSCF0049.RAF"
OUTPUT_PATH = "output/result.png"
TILE = 288*4
MARGIN = 12  # discarded ring around each tile; TILE - 2*MARGIN and MARGIN must be multiples of 6


@torch.no_grad()
def run_tiled(model, cfa: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Tiled inference with crop-discard margins; returns (3, H, W) linear RGB."""
    h, w = cfa.shape
    aligned = np.pad(cfa, ((dy, 0), (dx, 0)), mode="symmetric")
    ah, aw = aligned.shape
    stride = TILE - 2 * MARGIN
    gh = (ah + stride - 1) // stride * stride
    gw = (aw + stride - 1) // stride * stride
    big = np.pad(aligned, ((MARGIN, gh - ah + MARGIN), (MARGIN, gw - aw + MARGIN)), mode="edge")

    out = torch.zeros(3, gh, gw)
    for y in range(0, gh, stride):
        for x in range(0, gw, stride):
            tile = torch.from_numpy(big[y:y + TILE, x:x + TILE])[None, None].to(DEVICE)
            res = model(tile)[0, :, MARGIN:-MARGIN, MARGIN:-MARGIN]
            out[:, y:y + stride, x:x + stride] = res.cpu()
    return out[:, dy:dy + h, dx:dx + w].numpy()


if __name__ == "__main__":
    model = PackedXTransNet(XTRANS_PATTERN, width=32, depth=8).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    print(f"Decoding {RAF_PATH}...")
    with rawler_py.RawImage.open(RAF_PATH) as img:
        print(f"  {img.make} {img.model} | {img.width}x{img.height}")

        if img.active_area is not None:
            left, top, w, h = img.active_area
        else:
            left, top, w, h = 0, 0, img.width, img.height
        raw_data = img.raw_data()[top:top + h, left:left + w].astype(np.float32)

        black = float(img.blacklevel["levels"][0])
        white = float(img.whitelevel[0])

        wb = np.array(img.wb_coeffs[:3], dtype=np.float32)
        if np.isnan(wb[1]) or wb[1] == 0:
            wb[1] = 1.0
        wb /= wb[1]

        period = 6 if len(img.cfa_pattern) == 36 else 2
        char_to_color = {"R": 0, "G": 1, "B": 2}
        pattern_grid = np.zeros((period, period), dtype=np.int32)
        for y in range(period):
            for x in range(period):
                char = img.cfa_pattern[((y + top) % period) * period + ((x + left) % period)]
                pattern_grid[y, x] = char_to_color[char]

        canonical = XTRANS_PATTERN if period == 6 else BAYER_PATTERN
        shift = (0, 0)
        for sy in range(period):
            for sx in range(period):
                if np.array_equal(np.roll(canonical, (-sy, -sx), axis=(0, 1)), pattern_grid):
                    shift = (sy, sx)
        print(f"  CFA shift: dy={shift[0]} dx={shift[1]}")

        cfa = np.clip((raw_data - black) / (white - black), 0.0, 1.2)
        tiled = np.tile(pattern_grid, ((h + period - 1) // period, (w + period - 1) // period))[:h, :w]
        cfa = cfa * wb[tiled]

        print(f"Running inference on {w}x{h}...")
        t0 = time.perf_counter()
        rgb = run_tiled(model, cfa, shift[0], shift[1])
        print(f"  {time.perf_counter() - t0:.2f}s")

        rgb = apply_raw_processing(rgb, img)

    out = np.clip(np.round(np.transpose(rgb, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
    Path(OUTPUT_PATH).parent.mkdir(exist_ok=True)
    Image.fromarray(out, "RGB").save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
