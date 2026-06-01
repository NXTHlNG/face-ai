from __future__ import annotations

import math


def _lab_f(t: float) -> float:
    d = 6.0 / 29.0
    if t > d**3:
        return t ** (1.0 / 3.0)
    return t / (3.0 * d**2) + 4.0 / 29.0


def rgb_to_lab(r: float, g: float, b: float) -> tuple[float, float, float]:
    """sRGB 0-255 -> CIELAB."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = lin(r), lin(g), lin(b)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883
    fx, fy, fz = _lab_f(x), _lab_f(y), _lab_f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_star = 200.0 * (fy - fz)
    return L, a, b_star


def opencv_lab_to_cielab(L: float, a: float, b: float) -> tuple[float, float, float]:
    """OpenCV LAB (L 0-255, a/b 0-255 centered at 128) -> CIELAB."""
    return (L * 100.0 / 255.0, a - 128.0, b - 128.0)


def delta_e_cie2000(
    lab1: tuple[float, float, float],
    lab2: tuple[float, float, float],
) -> float:
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1.0 - math.sqrt(C_bar**7 / (C_bar**7 + 25.0**7 + 1e-12)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0
    dLp = L2 - L1
    dCp = C2p - C1p
    if C1p * C2p < 1e-12:
        dHp = 0.0
    elif abs(h2p - h1p) <= 180.0:
        dHp = h2p - h1p
    elif h2p <= h1p:
        dHp = h2p - h1p + 360.0
    else:
        dHp = h2p - h1p - 360.0
    dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dHp / 2.0))
    L_bar = (L1 + L2) / 2.0
    C_bar_p = (C1p + C2p) / 2.0
    if C1p * C2p < 1e-12:
        h_bar_p = h1p + h2p
    elif abs(h1p - h2p) <= 180.0:
        h_bar_p = (h1p + h2p) / 2.0
    elif h1p + h2p < 360.0:
        h_bar_p = (h1p + h2p + 360.0) / 2.0
    else:
        h_bar_p = (h1p + h2p - 360.0) / 2.0
    T = (
        1.0
        - 0.17 * math.cos(math.radians(h_bar_p - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * h_bar_p))
        + 0.32 * math.cos(math.radians(3.0 * h_bar_p + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * h_bar_p - 63.0))
    )
    Sl = 1.0 + 0.015 * (L_bar - 50.0) ** 2 / math.sqrt(20.0 + (L_bar - 50.0) ** 2)
    Sc = 1.0 + 0.045 * C_bar_p
    Sh = 1.0 + 0.015 * C_bar_p * T
    Rt = (
        -2.0
        * math.sqrt(C_bar_p**7 / (C_bar_p**7 + 25.0**7 + 1e-12))
        * math.sin(math.radians(60.0 * math.exp(-((h_bar_p - 275.0) / 25.0) ** 2)))
    )
    return math.sqrt(
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )


def nearest_palette_class(
    lab: tuple[float, float, float],
    palette: dict[str, tuple[float, float, float]],
) -> tuple[str, float, dict[str, float]]:
    scores: dict[str, float] = {}
    for name, ref in palette.items():
        scores[name] = delta_e_cie2000(lab, ref)
    best = min(scores, key=scores.get)
    return best, scores[best], scores
