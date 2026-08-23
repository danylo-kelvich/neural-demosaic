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
from scripts.export_coreml import ExportModel
from src.utils import apply_raw_processing

CKPT = "weights/packed_5183_3208.pt"
MLPKG = "weights/packed_coreml.mlpackage"
WIDTH, DEPTH = 32, 8
RAF_PATH = "samples/DSCF0049.RAF"
OUTPUT_PATH = "output/result_coreml.png"
TILE = 288
MARGIN = 12


def load_mlmodel():
    import coremltools as ct

    if not Path(MLPKG).exists() or Path(MLPKG).stat().st_mtime < Path(CKPT).stat().st_mtime:
        print(f"Converting {CKPT} -> {MLPKG}...")
        model = PackedXTransNet(XTRANS_PATTERN, width=WIDTH, depth=DEPTH)
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.eval()
        wrapped = ExportModel(model, TILE).eval()
        traced = torch.jit.trace(wrapped, torch.rand(1, 1, TILE, TILE))
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="mosaic", shape=(1, 1, TILE, TILE), dtype=np.float16)],
            outputs=[ct.TensorType(name="rgb", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS26,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )
        mlmodel.save(MLPKG)
    return ct.models.MLModel(MLPKG, compute_units=ct.ComputeUnit.CPU_AND_NE)


def run_tiled(mlmodel, cfa: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Tiled CoreML inference with crop-discard margins; returns (3, H, W) linear RGB."""
    h, w = cfa.shape
    aligned = np.pad(cfa, ((dy, 0), (dx, 0)), mode="symmetric")
    ah, aw = aligned.shape
    stride = TILE - 2 * MARGIN
    gh = (ah + stride - 1) // stride * stride
    gw = (aw + stride - 1) // stride * stride
    big = np.pad(aligned, ((MARGIN, gh - ah + MARGIN), (MARGIN, gw - aw + MARGIN)),
                 mode="edge").astype(np.float16)

    out = np.zeros((3, gh, gw), dtype=np.float32)
    for y in range(0, gh, stride):
        for x in range(0, gw, stride):
            tile = big[y:y + TILE, x:x + TILE][None, None]
            rgb = mlmodel.predict({"mosaic": tile})["rgb"]
            out[:, y:y + stride, x:x + stride] = rgb[0, :, MARGIN:-MARGIN, MARGIN:-MARGIN]
    return out[:, dy:dy + h, dx:dx + w]


if __name__ == "__main__":
    mlmodel = load_mlmodel()

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

        print(f"Running CoreML inference on {w}x{h}...")
        t0 = time.perf_counter()
        rgb = run_tiled(mlmodel, cfa, shift[0], shift[1])
        elapsed = time.perf_counter() - t0
        print(f"  {elapsed:.2f}s ({w * h / elapsed / 1e6:.1f} MP/s)")

        apply_raw_processing(rgb, img)

    out = np.clip(np.round(np.transpose(rgb, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
    Path(OUTPUT_PATH).parent.mkdir(exist_ok=True)
    Image.fromarray(out, "RGB").save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
