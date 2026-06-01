"""AUA Capstone–style color heuristics (hue-trim skin, lip brightness tiers, multi-zone chroma)."""

from __future__ import annotations

import cv2
import numpy as np

from app.backends.color.lab_utils import mean_lab, median_lab

# OpenCV HSV hue 0–179 (same scale as wrist skin mask in this project).
SKIN_HUE_TRIM_LO = 13
SKIN_HUE_TRIM_HI = 24

# Colorinsight lip pre-filter (red channel floor, blue ceiling).
LIP_R_MIN = 97
LIP_B_MAX = 227
LIP_LUM_MIN = 50.0

# Exclude specular highlights and deep shadows on skin (OpenCV L percentiles).
SKIN_L_SHADOW_PCT = 18.0
SKIN_L_HIGHLIGHT_PCT = 88.0

# Hair: mid-bright band avoids roots/shadows under bangs.
HAIR_LUM_PCT_LO = 55.0
HAIR_LUM_PCT_HI = 88.0


def _luminance_rgb(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, 0].astype(np.float32)
    g = rgb[:, 1].astype(np.float32)
    b = rgb[:, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def skin_trim_pixels(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
    *,
    hue_lo: int = SKIN_HUE_TRIM_LO,
    hue_hi: int = SKIN_HUE_TRIM_HI,
    min_pixels: int = 24,
    shadow_pct: float = SKIN_L_SHADOW_PCT,
    highlight_pct: float = SKIN_L_HIGHLIGHT_PCT,
) -> np.ndarray | None:
    """RGB pixels: hue-trimmed skin in mid-L band (no highlights, no deep shadows)."""
    m = skin_mask > 127
    if not np.any(m):
        return None
    pixels = image_rgb[m].reshape(-1, 3)
    if pixels.shape[0] < 8:
        return None

    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    trim = (hue >= hue_lo) & (hue <= hue_hi)
    n_trim = int(np.count_nonzero(trim))
    base = pixels[trim] if n_trim >= min_pixels else pixels

    lab = cv2.cvtColor(base.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    l_vals = lab[:, 0].astype(np.float32)
    lo_cut = float(np.percentile(l_vals, shadow_pct))
    hi_cut = float(np.percentile(l_vals, highlight_pct))
    if hi_cut <= lo_cut + 4.0:
        return base
    mid = (l_vals >= lo_cut) & (l_vals <= hi_cut)
    if int(np.count_nonzero(mid)) >= min_pixels:
        return base[mid]
    return base


def skin_lab_hue_trim(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
    *,
    hue_lo: int = SKIN_HUE_TRIM_LO,
    hue_hi: int = SKIN_HUE_TRIM_HI,
    min_pixels: int = 24,
) -> tuple[float, float, float] | None:
    """Median LAB on hue-trimmed skin without highlight bias (AUA Method 4)."""
    trimmed = skin_trim_pixels(
        image_rgb,
        skin_mask,
        hue_lo=hue_lo,
        hue_hi=hue_hi,
        min_pixels=min_pixels,
    )
    if trimmed is None:
        return median_lab(image_rgb, skin_mask) or mean_lab(image_rgb, skin_mask)
    if trimmed.shape[0] < 8:
        return median_lab(image_rgb, skin_mask) or mean_lab(image_rgb, skin_mask)

    lab = cv2.cvtColor(trimmed.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.median(lab[:, i])) for i in range(3))


def hair_lab_luminance_trim(
    image_rgb: np.ndarray,
    hair_mask: np.ndarray,
    skin_mask: np.ndarray | None = None,
    *,
    lum_lo_pct: float = HAIR_LUM_PCT_LO,
    lum_hi_pct: float = HAIR_LUM_PCT_HI,
    min_pixels: int = 24,
) -> tuple[float, float, float] | None:
    """Median LAB on mid-bright hair pixels, excluding skin overlap and deep shadows."""
    m = hair_mask > 127
    if skin_mask is not None and np.any(skin_mask > 127):
        m = m & (skin_mask <= 127)
    if not np.any(m):
        m = hair_mask > 127
    if not np.any(m):
        return None

    roi = image_rgb[m].reshape(-1, 3)
    if roi.shape[0] < 8:
        return median_lab(image_rgb, hair_mask) or mean_lab(image_rgb, hair_mask)

    gray = cv2.cvtColor(roi.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).reshape(-1)
    lo = float(np.percentile(gray, lum_lo_pct))
    hi = float(np.percentile(gray, lum_hi_pct))
    if hi <= lo + 2.0:
        return median_lab(image_rgb, hair_mask) or mean_lab(image_rgb, hair_mask)

    keep = (gray >= lo) & (gray <= hi)
    if int(np.count_nonzero(keep)) < min_pixels:
        return median_lab(image_rgb, hair_mask) or mean_lab(image_rgb, hair_mask)

    sel = roi[keep]
    lab = cv2.cvtColor(sel.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.median(lab[:, i])) for i in range(3))


def brow_lab_from_mask(
    image_rgb: np.ndarray,
    brow_mask: np.ndarray,
    hair_mask: np.ndarray | None = None,
    *,
    min_pixels: int = 12,
) -> tuple[float, float, float] | None:
    """Median LAB on brows, excluding hair overlap (e.g. bangs)."""
    m = brow_mask > 127
    if hair_mask is not None and np.any(hair_mask > 127):
        m = m & (hair_mask <= 127)
    if not np.any(m):
        m = brow_mask > 127
    if not np.any(m):
        return None
    out_mask = np.zeros_like(brow_mask, dtype=np.uint8)
    out_mask[m] = 255
    if np.count_nonzero(m) < min_pixels:
        return median_lab(image_rgb, brow_mask) or mean_lab(image_rgb, brow_mask)
    return median_lab(image_rgb, out_mask)


def lip_lab_brightness_clusters(
    image_rgb: np.ndarray,
    lip_mask: np.ndarray,
    *,
    min_pixels: int = 12,
) -> tuple[float, float, float] | None:
    """Three brightness tiers on lip mask; median LAB of middle tier (AUA Ch.2)."""
    m = lip_mask > 127
    if not np.any(m):
        return None
    pixels = image_rgb[m].reshape(-1, 3)
    if pixels.shape[0] < min_pixels:
        return None

    lum = _luminance_rgb(pixels)
    keep = (
        (pixels[:, 0] >= LIP_R_MIN)
        & (pixels[:, 2] <= LIP_B_MAX)
        & (lum >= LIP_LUM_MIN)
    )
    if int(np.count_nonzero(keep)) < min_pixels:
        keep = lum >= LIP_LUM_MIN
    if int(np.count_nonzero(keep)) < 8:
        return median_lab(image_rgb, lip_mask)

    sel = pixels[keep]
    sel_lum = lum[keep]
    order = np.argsort(sel_lum)
    sel = sel[order]
    n = sel.shape[0]
    third = max(1, n // 3)
    mid = sel[third : 2 * third]
    if mid.shape[0] < 4:
        mid = sel[max(0, n // 3 - 1) : min(n, 2 * n // 3 + 1)]

    lab = cv2.cvtColor(mid.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return tuple(float(np.median(lab[:, i])) for i in range(3))


def chroma_from_ab(ab: tuple[float, float]) -> float:
    return float(np.hypot(ab[0], ab[1]))


def aggregate_chroma(
    skin_ab: tuple[float, float],
    eyes_ab: tuple[float, float] | None = None,
    lips_ab: tuple[float, float] | None = None,
) -> float:
    """AUA: chroma = mean(skin, eyes, lips) — not skin alone."""
    parts = [chroma_from_ab(skin_ab)]
    if eyes_ab is not None:
        parts.append(chroma_from_ab(eyes_ab))
    if lips_ab is not None:
        parts.append(chroma_from_ab(lips_ab))
    return float(np.mean(parts))
