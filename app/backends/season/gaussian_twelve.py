from __future__ import annotations

import numpy as np

from app.backends.season.season_maps import SIXTEEN_TO_TWELVE, TWELVE_TO_FOUR

# Temporarily not wired into extract/ensemble; kept for future re-enable.


def _gauss_1d(x: float, mu: float, sigma: float) -> float:
    s = max(sigma, 0.45)
    return float(np.exp(-0.5 * ((x - mu) / s) ** 2))


def _clarity_axis(contrast_bucket: str, value_contrast_index: float, skin_chroma: float) -> tuple[float, float]:
    x = min(1.0, max(0.0, value_contrast_index / 74.0))
    if contrast_bucket == "high":
        x = min(1.0, x + 0.19)
    elif contrast_bucket == "low":
        x = max(0.0, x - 0.17)
    x = min(1.0, x + max(0.0, skin_chroma - 15.5) / 52.0)
    clear = min(1.0, max(0.0, x))
    soft = min(1.0, max(0.0, 1.0 - clear))
    if skin_chroma < 13.5 and contrast_bucket == "low":
        soft = min(1.0, soft + 0.14)
        clear = max(0.0, clear - 0.14)
    return clear, soft


def _warmth_axis_signed(
    undertone: str,
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
) -> float:
    if undertone == "warm":
        return 1.0
    if undertone == "cool":
        return -1.0
    acc = 0.0
    if hair_ab:
        acc += hair_ab[1]
    if iris_ab:
        acc += iris_ab[1]
    return float(np.clip(np.tanh(acc / 17.5), -1.0, 1.0))


def _depth_axis_signed(depth: str, skin_L: float) -> float:
    if depth == "light":
        return -1.0
    if depth == "deep":
        return 1.0
    t = (skin_L - 56.0) / (76.0 - 56.0)
    t = min(1.0, max(0.0, t))
    return float(1.0 - 2.0 * t)


def classify_gaussian_twelve(
    undertone: str,
    depth: str,
    skin_chroma: float,
    contrast_bucket: str,
    value_contrast_index: float,
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
    skin_L: float,
) -> dict:
    clear, soft = _clarity_axis(contrast_bucket, value_contrast_index, skin_chroma)
    w = _warmth_axis_signed(undertone, hair_ab, iris_ab)
    d = _depth_axis_signed(depth, skin_L)
    wp, wn = max(0.0, w), max(0.0, -w)
    lp, ln = max(0.0, -d), max(0.0, d)
    low_c = contrast_bucket == "low"
    high_c = contrast_bucket == "high"
    vci_n = min(1.35, value_contrast_index / 76.0)
    chrom_n = min(1.0, skin_chroma / 36.0)
    gl = _gauss_1d(skin_L, 73.8, 5.2)
    gm = _gauss_1d(skin_L, 64.0, 7.8)
    gd = _gauss_1d(skin_L, 52.8, 6.6)
    win_mix = wn * (0.74 * ln + 0.26 * max(lp, 0.2))

    scores: dict[str, float] = {
        "light_spring": wp * lp * soft * (0.52 + 0.48 * gl) * (1.05 + 0.08 * (1.0 - chrom_n)),
        "bright_spring": wp * lp * clear * (0.42 + 0.58 * vci_n) * (1.08 if high_c else 1.0),
        "true_spring": wp * lp * (0.4 + 0.36 * clear + 0.38 * soft) * (0.85 + 0.35 * gm),
        "light_summer": wn * lp * soft * (0.5 + 0.5 * gl) * (1.06 - 0.12 * chrom_n),
        "soft_summer": wn * lp * (0.46 + 0.54 * soft) * (1.14 if low_c else 1.0) * (1.02 + 0.06 * (1.0 - vci_n)),
        "true_summer": wn * lp * (0.38 + 0.34 * clear + 0.44 * soft) * (0.88 + 0.32 * gm),
        "soft_autumn": wp * ln * (0.44 + 0.56 * soft) * (1.08 if low_c else 1.0),
        "true_autumn": wp * ln * (0.42 + 0.34 * clear + 0.38 * soft) * (0.88 + 0.28 * gm),
        "deep_autumn": wp * ln * (0.4 + 0.6 * gd) * (0.52 + 0.48 * ln) * (0.98 + 0.16 * soft),
        "bright_winter": win_mix * clear * (0.45 + 0.55 * vci_n)
        + wn * lp * clear * 0.26
        + (0.12 * wn * clear * high_c),
        "true_winter": wn * ln * (0.42 + 0.38 * clear + 0.34 * soft) * (0.86 + 0.3 * gm),
        "deep_winter": wn * ln * (0.42 + 0.58 * gd) * (0.48 + 0.52 * clear),
    }

    total = sum(scores.values()) + 1e-9
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_twelve, best_val = ranked[0]
    second_val = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_val - second_val
    rel = margin / total
    conf = 0.34 + min(0.48, rel * 3.5 + margin * 3.0)
    if undertone == "neutral":
        conf *= 0.82
    if depth == "medium":
        conf *= 0.88
    if best_val / total < 0.14:
        conf *= 0.68
    conf = float(min(0.88, max(0.22, conf)))
    if total < 1e-9 or best_val < 0.028:
        return {
            "seasonal_twelve": "unknown",
            "seasonal_guess": "unknown",
            "confidence": max(0.17, conf * 0.48),
            "scores": scores,
            "top_k_twelve": [],
            "top_k_four": [],
        }

    probs_12 = [(k, v / total) for k, v in ranked]
    four_scores: dict[str, float] = {}
    for k, p in probs_12:
        parent = TWELVE_TO_FOUR.get(k, "unknown")
        four_scores[parent] = four_scores.get(parent, 0.0) + p
    ranked_four = sorted(four_scores.items(), key=lambda x: -x[1])

    return {
        "seasonal_twelve": best_twelve,
        "seasonal_guess": TWELVE_TO_FOUR.get(best_twelve, "unknown"),
        "confidence": conf,
        "scores": scores,
        "top_k_twelve": [{"subtype": k, "probability": round(p, 4)} for k, p in probs_12[:3]],
        "top_k_four": [{"season": k, "probability": round(p, 4)} for k, p in ranked_four[:2]],
    }
