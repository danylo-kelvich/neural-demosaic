import sys
from pathlib import Path

import numpy as np
import rawler_py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.demosaic_opencl import MarkesteijnOpenCLDemosaicer
from src.utils import apply_raw_processing
import os
os.environ['PYOPENCL_CTX'] = '0'

RAF_PATH = "samples/DSCF0049.RAF"
OUTPUT_PATH = "output/result_markesteijn.png"
KERNEL_PATH = "src/kernels/demosaic_markesteijn.cl"


if __name__ == "__main__":
    demosaicer = MarkesteijnOpenCLDemosaicer(kernel_path=KERNEL_PATH)

    print(f"Decoding {RAF_PATH}...")
    with rawler_py.RawImage.open(RAF_PATH) as img:
        print(f"  {img.make} {img.model} | {img.width}x{img.height}")

        if img.active_area is not None:
            left, top, w, h = img.active_area
        else:
            left, top, w, h = 0, 0, img.width, img.height

        print(f"  active {w}x{h} (top={top}, left={left})")
        raw_data = img.raw_data()[top:top+h, left:left+w].astype(np.float32)

        black = float(img.blacklevel["levels"][0])
        white = float(img.whitelevel[0])

        wb = np.array(img.wb_coeffs[:3], dtype=np.float32)
        if np.isnan(wb[1]) or wb[1] == 0:
            wb[1] = 1.0
        wb /= wb[1]
        print(f"  WB (G=1): R={wb[0]:.3f} B={wb[2]:.3f}")

        period = 6 if len(img.cfa_pattern) == 36 else 2
        char_to_color = {"R": 0, "G": 1, "B": 2}
        pattern_grid = np.zeros((period, period), dtype=np.uint8)
        for y in range(period):
            for x in range(period):
                char = img.cfa_pattern[((y + top) % period) * period + ((x + left) % period)]
                pattern_grid[y, x] = char_to_color[char]

        cfa = np.clip((raw_data - black) / (white - black), 0.0, 1.2)

        tiled_colors = np.tile(pattern_grid, ((h + period - 1) // period, (w + period - 1) // period))[:h, :w]
        wb_mask = wb[tiled_colors]
        cfa = cfa * wb_mask

        print(f"Running Markesteijn demosaic on {w}x{h}...")
        rgb_hwc = demosaicer.demosaic(raw_image=cfa, xtrans_pattern=pattern_grid, passes=1)
        rgb = np.transpose(rgb_hwc, (2, 0, 1)).astype(np.float32)

        rgb = apply_raw_processing(rgb, img)

    out = np.clip(np.round(np.transpose(rgb, (1, 2, 0)) * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(out, 'RGB').save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
