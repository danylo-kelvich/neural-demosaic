import numpy as np
from rawler_py.rawler_py import RawImage

def apply_color_matrix(rgb: np.ndarray, xyz_to_cam: np.ndarray) -> np.ndarray:
    """Convert camera RGB to sRGB via row-normalized color matrix."""
    XYZ_TO_SRGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    srgb_to_xyz = np.linalg.inv(XYZ_TO_SRGB)
    srgb_to_cam = xyz_to_cam @ srgb_to_xyz
    
    # Row normalize srgb_to_cam
    row_sums = srgb_to_cam.sum(axis=1, keepdims=True)
    row_sums = np.where(np.abs(row_sums) > 1e-8, row_sums, 1.0)
    srgb_to_cam /= row_sums
    
    cam_to_srgb = np.linalg.inv(srgb_to_cam)
    h, w = rgb.shape[1], rgb.shape[2]
    return np.maximum(cam_to_srgb @ rgb.reshape(3, -1), 0).reshape(3, h, w)


def apply_aces(rgb: np.ndarray, exposure: float = 2.0) -> np.ndarray:
    """Narkowicz ACES tonemapping with exposure pre-multiply."""
    x = rgb * exposure
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


def apply_srgb_gamma(rgb: np.ndarray) -> np.ndarray:
    """Apply piecewise sRGB gamma curve."""
    rgb = np.clip(rgb, 0, 1)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)


def apply_orientation(rgb: np.ndarray, orientation: str) -> np.ndarray:
    """Apply orientation to (3, H, W) rgb array based on rawler's contract."""
    mapping = {
        "Normal": (False, False, False),
        "ReverseYaw": (False, True, False),
        "Rotate180": (False, True, True),
        "ReverseRoll": (False, False, True),
        "ReversePitch": (True, False, False),   # MH + R270 = pure transpose
        "Rotate90": (True, True, False),         # R90 CW = transpose + hflip cols
        "MirrorHorizontalAndRotate90": (True, True, True),
        "Rotate270": (True, False, True),        # R270 CW = transpose + vflip rows
    }
    transpose, hflip, vflip = mapping.get(orientation, (False, False, False))
    if transpose:
        rgb = np.transpose(rgb, (0, 2, 1))  # (3,H,W) -> (3,W,H): axis1=rows(W), axis2=cols(H)
        if hflip:
            rgb = np.flip(rgb, axis=2)  # flip cols (H dim) = horizontal
        if vflip:
            rgb = np.flip(rgb, axis=1)  # flip rows (W dim) = vertical
    else:
        if hflip:
            rgb = np.flip(rgb, axis=2)
        if vflip:
            rgb = np.flip(rgb, axis=1)
    return rgb


def apply_raw_processing(rgb: np.ndarray, img: RawImage) -> np.ndarray:
    xyz_to_cam = np.array(img.xyz_to_cam[:3], dtype=np.float32)
    if not np.any(xyz_to_cam):
        if "D65" in img.color_matrix:
            xyz_to_cam = np.array(img.color_matrix["D65"], dtype=np.float32).reshape(3, 3)
        elif img.color_matrix:
            first_key = list(img.color_matrix.keys())[0]
            xyz_to_cam = np.array(img.color_matrix[first_key], dtype=np.float32).reshape(3, 3)
        else:
            xyz_to_cam = np.eye(3, dtype=np.float32)

    rgb = apply_color_matrix(rgb, xyz_to_cam)
    rgb = apply_aces(rgb)
    rgb = apply_srgb_gamma(rgb)
    rgb = apply_orientation(rgb, img.orientation)

    return rgb