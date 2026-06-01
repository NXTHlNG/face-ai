"""OpenCV LAB → AUA Capstone (Khachatryan et al. 2025) Munsell axis scales.

The capstone normalizes four factors to integers 1–5 before ``season_lookup_16.json``.
Our pipeline extracts colors in OpenCV LAB (L, a, b uint8-style: L∈[0,255],
a*, b* centered at 128). This module converts those measurements to the capstone
units, then applies the same bin thresholds as their Java ``normalize*`` methods.

Mapping summary
---------------
Chroma (0–1):
  ``C_lab = hypot(a*−128, b*−128)`` averaged over skin, eyes, lips.
  ``chroma_aua = min(1, C_lab / 40)`` — divisor calibrated so typical face
  chroma ~8–32 maps to AUA ~0.2–0.8 (muted → vivid).

Value (0–1):
  ``value_aua = composite_L / 255`` where
  ``composite_L = 0.7·skin_L + 0.15·eyes_L + 0.15·hair_L`` (same weights as capstone).

Contrast (0–21):
  ``contrast_aua = (L_max + 0.05) / (L_min + 0.05)`` on OpenCV L for skin, hair, eyes
  (capstone uses luminance min/max across the same zones; high ratio ⇒ high contrast).

Undertone (1–5) is still derived from fused warm/cool/neutral, not skin HSV hue.
"""

from __future__ import annotations

# Typical vivid face chroma in OpenCV a*b* units ≈ 32–40 → AUA 0.8–1.0.
AUA_CHROMA_LAB_DIVISOR = 40.0

# OpenCV L channel full scale (matches capstone treating value as 0–1 brightness).
AUA_VALUE_L_SCALE = 255.0


def lab_chroma_to_aua(chroma_lab: float) -> float:
    """Map mean LAB chroma (OpenCV a*b* units) to AUA intensity 0–1."""
    return min(1.0, max(0.0, float(chroma_lab) / AUA_CHROMA_LAB_DIVISOR))


def opencv_L_to_aua_value(L: float) -> float:
    """Map OpenCV LAB L (0–255) to AUA value brightness 0–1."""
    return min(1.0, max(0.0, float(L) / AUA_VALUE_L_SCALE))


def lab_luminance_contrast_aua(L_min: float, L_max: float) -> float:
    """Capstone contrast metric on OpenCV L; typical range ~1–21."""
    lo = min(float(L_min), float(L_max))
    hi = max(float(L_min), float(L_max))
    return (hi + 0.05) / (lo + 0.05)


def normalize_chroma_aua(chroma_0_1: float) -> int:
    """AUA Capstone ``normalizeChroma`` — thresholds 0.2 / 0.4 / 0.6 / 0.8."""
    c = float(chroma_0_1)
    if c < 0.2:
        return 1
    if c < 0.4:
        return 2
    if c < 0.6:
        return 3
    if c < 0.8:
        return 4
    return 5


def normalize_value_aua(value_0_1: float) -> int:
    """AUA Capstone ``normalizeValue`` — thresholds 0.2 / 0.4 / 0.6 / 0.8."""
    v = float(value_0_1)
    if v < 0.2:
        return 1
    if v < 0.4:
        return 2
    if v < 0.6:
        return 3
    if v < 0.8:
        return 4
    return 5


def normalize_contrast_aua(contrast_0_21: float) -> int:
    """AUA Capstone ``normalizeContrast`` — thresholds 5 / 9 / 13 / 17 / 21."""
    c = float(contrast_0_21)
    if c < 5.0:
        return 1
    if c < 9.0:
        return 2
    if c < 13.0:
        return 3
    if c < 17.0:
        return 4
    return 5
