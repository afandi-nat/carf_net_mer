"""Geometric normalisation, optical flow and optical strain.

These are the building blocks of the CARF-Net input tensor. Every scale here is
fixed in pixels of the aligned 224x224 face: no statistic is ever computed from
a single clip, so motion magnitude stays comparable across clips, subjects and
LOSO folds.
"""

import os

import cv2
import numpy as np
from PIL import Image

SIZE = 224

# Canonical template in relative coordinates. Eyes at 0.315 and mouth at 0.780
# put the h//2 split line around the nose bridge: below the eyes, above the
# mouth. The two-branch architecture depends on that line.
CANON = np.float32([
    [0.315 * SIZE, 0.315 * SIZE],   # left eye centre
    [0.685 * SIZE, 0.315 * SIZE],   # right eye centre
    [0.500 * SIZE, 0.780 * SIZE],   # mouth centre
])

LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))
HULL_IDX = list(range(0, 27))       # jaw line and brows

ECC_MOTION = {"affine": cv2.MOTION_AFFINE, "euclidean": cv2.MOTION_EUCLIDEAN}


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #
def similarity_matrix(pts):
    """2x3 similarity mapping eye centres and mouth centre onto the template.

    Because the eye-to-eye distance of the template is fixed, this is an
    inter-ocular-distance (IOD) normalisation: scale, rotation and translation
    are removed, and only expression motion survives in the flow field.
    """
    src = np.float32([
        pts[LEFT_EYE].mean(axis=0),
        pts[RIGHT_EYE].mean(axis=0),
        pts[MOUTH].mean(axis=0),
    ])
    matrix, _ = cv2.estimateAffinePartial2D(src, CANON, method=cv2.LMEDS,
                                            refineIters=20)
    if matrix is None:
        scale = SIZE / max(1.0, float(np.ptp(pts[:, 0])))
        matrix = np.float32([[scale, 0, 0], [0, scale, 0]])
    return matrix


def warp(img, matrix):
    return cv2.warpAffine(img, matrix, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)


def fallback_geometry(shape):
    """Transform and mask used when no landmarks are available: plain resize
    plus an elliptical face mask."""
    height, width = shape[:2]
    matrix = np.float32([[SIZE / width, 0, 0], [0, SIZE / height, 0]])
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.ellipse(mask, (SIZE // 2, SIZE // 2),
                (int(0.46 * SIZE), int(0.49 * SIZE)), 0, 0, 360, 1, -1)
    return matrix, mask.astype(bool)


def masks_from_landmarks(pts, matrix, feather):
    """Hard mask (for fitting) and feathered mask (for the output channels)."""
    ones = np.ones((len(pts), 1), dtype=np.float32)
    warped = (np.hstack([pts, ones]) @ matrix.T).astype(np.int32)
    hull = cv2.convexHull(warped[HULL_IDX])
    hard = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.fillConvexPoly(hard, hull, 1)
    hard = cv2.dilate(hard, np.ones((9, 9), np.uint8), iterations=1).astype(bool)
    return hard, feather_mask(hard, feather)


def feather_mask(hard, sigma):
    """Gaussian feathering of the hull edge; the interior stays exactly 1."""
    if sigma <= 0:
        return hard.astype(np.float32)
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigma)
    inner = cv2.erode(hard.astype(np.uint8), np.ones((5, 5), np.uint8), 1)
    return np.maximum(soft, inner).clip(0, 1)


def ecc_align(template, moving, mask, motion, max_shift=12.0):
    """Align `moving` onto `template` with an ECC affine fit restricted to the
    face mask. Returns the original image and None when ECC fails or returns an
    implausible transform, so a bad fit can never corrupt a clip."""
    warp_m = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp_m = cv2.findTransformECC(
            template.astype(np.float32), moving.astype(np.float32), warp_m,
            ECC_MOTION[motion], criteria, mask.astype(np.uint8), 5,
        )
    except cv2.error:
        return moving, None
    if (np.abs(warp_m[:, 2]).max() > max_shift
            or np.abs(warp_m[:, :2] - np.eye(2)).max() > 0.08):
        return moving, None
    return apply_ecc(moving, warp_m), warp_m


def apply_ecc(img, warp_m):
    if warp_m is None:
        return img
    return cv2.warpAffine(img, warp_m, (SIZE, SIZE),
                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REFLECT)


# --------------------------------------------------------------------------- #
# Optical flow
# --------------------------------------------------------------------------- #
def make_tvl1():
    if hasattr(cv2, "optflow") and hasattr(cv2.optflow, "DualTVL1OpticalFlow_create"):
        return cv2.optflow.DualTVL1OpticalFlow_create(scaleStep=0.5)
    raise SystemExit(
        "TV-L1 optical flow is unavailable. Install opencv-contrib-python "
        "(and make sure no other opencv package is installed alongside it)."
    )


def multiscale_flow(prev_gray, next_gray, scales=(1.0, 0.75), blur=5):
    """Average TV-L1 flow over several scales, then median-blur the components.

    The coarse scale captures larger displacements with less noise, the full
    scale keeps detail, and the median filter removes single-pixel spikes.
    """
    height, width = prev_gray.shape
    accumulator = np.zeros((height, width, 2), dtype=np.float32)

    for scale in scales:
        if scale == 1.0:
            a, b = prev_gray, next_gray
        else:
            size = (int(round(width * scale)), int(round(height * scale)))
            a = cv2.resize(prev_gray, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(next_gray, size, interpolation=cv2.INTER_AREA)
        flow = make_tvl1().calc(a, b, None)
        if scale != 1.0:
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow /= scale   # rescale displacements back to full-resolution pixels
        accumulator += flow

    flow = accumulator / len(scales)
    if blur and blur >= 3:
        # medianBlur on CV_32F only accepts ksize 3 and 5, and that constraint
        # has changed between OpenCV versions; fall back to a Gaussian.
        try:
            flow[..., 0] = cv2.medianBlur(flow[..., 0], blur)
            flow[..., 1] = cv2.medianBlur(flow[..., 1], blur)
        except cv2.error:
            flow = cv2.GaussianBlur(flow, (blur, blur), 0)
    return flow


def remove_global_affine(flow, mask, iters=3):
    """Subtract rigid and affine head motion from the flow field.

    The model u = a0 + a1·x + a2·y (and likewise for v) covers head
    translation, rotation, scaling and shear. Cauchy weighting makes
    large-residual pixels -- which are exactly the local muscle motions worth
    keeping -- unable to drag the fit.
    """
    height, width = flow.shape[:2]
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return flow, np.zeros(6, dtype=np.float32)

    xn = xs / width - 0.5
    yn = ys / height - 0.5
    design = np.stack([np.ones_like(xn), xn, yn], axis=1).astype(np.float32)

    u, v = flow[ys, xs, 0], flow[ys, xs, 1]
    weight = np.ones_like(xn, dtype=np.float32)
    coef_u = coef_v = np.zeros(3, dtype=np.float32)

    for _ in range(iters):
        sqrt_w = np.sqrt(weight)[:, None]
        coef_u, *_ = np.linalg.lstsq(design * sqrt_w, u * sqrt_w[:, 0], rcond=None)
        coef_v, *_ = np.linalg.lstsq(design * sqrt_w, v * sqrt_w[:, 0], rcond=None)
        residual = np.hypot(u - design @ coef_u, v - design @ coef_v)
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        weight = 1.0 / (1.0 + (residual / (2.385 * scale)) ** 2)

    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)
    grid_x = grid_x / width - 0.5
    grid_y = grid_y / height - 0.5
    full = np.stack([np.ones_like(grid_x), grid_x, grid_y], axis=-1)

    out = flow.copy()
    out[..., 0] -= full @ coef_u
    out[..., 1] -= full @ coef_v
    return out, np.concatenate([coef_u, coef_v]).astype(np.float32)


def strain_from_flow(flow):
    """Optical strain magnitude, in the same pixel units as the flow field."""
    u, v = flow[..., 0], flow[..., 1]
    u_x = np.gradient(u, axis=1)
    u_y = np.gradient(u, axis=0)
    v_x = np.gradient(v, axis=1)
    v_y = np.gradient(v, axis=0)
    e_xy = 0.5 * (u_y + v_x)
    return np.sqrt(u_x ** 2 + 2 * e_xy ** 2 + v_y ** 2)


# --------------------------------------------------------------------------- #
def read_gray_window(base, number, half_window):
    """Temporal average of grayscale frames around a frame number.

    Missing neighbours are simply skipped, so the pipeline still runs when only
    the three annotated frames were exported.
    """
    frames = []
    for offset in range(-half_window, half_window + 1):
        path = f"{base}/img{number + offset}.jpg"
        if os.path.isfile(path):
            frames.append(cv2.cvtColor(np.array(Image.open(path).convert("RGB")),
                                       cv2.COLOR_RGB2GRAY))
    if not frames:
        return None, 0
    shape = frames[0].shape
    frames = [f for f in frames if f.shape == shape]
    return np.mean(frames, axis=0).astype(np.float32), len(frames)
