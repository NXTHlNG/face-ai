"""Park et al. IMCOM'18 personal color rules (§3.2.4).

Undertone: skin-only Lab — cool if a* > b*, else warm.
Season (4): warm → spring/autumn, cool → summer/winter; bright vs muted via
mean pairwise |ΔL| among skin, hair, iris (CIELAB L 0–100) vs threshold 13.
"""

from __future__ import annotations

from app.backends.season.season_maps import TWELVE_TO_FOUR
from app.backends.color.undertone import park_undertone_from_ab

FOUR_SEASONS = ("spring", "summer", "autumn", "winter")

# Canonical 12-subtype labels for API compatibility (Park outputs 4 seasons only).
FOUR_TO_TWELVE: dict[str, str] = {
    "spring": "true_spring",
    "summer": "soft_summer",
    "autumn": "true_autumn",
    "winter": "true_winter",
}


def opencv_L_to_cielab(L_opencv: float) -> float:
    """OpenCV L channel (0–255) → CIELAB L* (0–100)."""
    return float(L_opencv) * 100.0 / 255.0


def park_undertone(skin_ab: tuple[float, float]) -> str:
    """Skin a* vs b* (CIELAB, centered at 0) with fair pink-cool heuristic."""
    return park_undertone_from_ab(skin_ab)


def park_mean_contrast_L(
    skin_L: float,
    hair_L: float | None,
    iris_L: float | None,
    *,
    use_cielab_scale: bool = True,
) -> tuple[float, list[str]]:
    """Mean |ΔL| over available pairs among skin, hair, iris (Park's three regions)."""
    def _L(x: float) -> float:
        return opencv_L_to_cielab(x) if use_cielab_scale else float(x)

    vals: dict[str, float] = {"skin": _L(skin_L)}
    if hair_L is not None:
        vals["hair"] = _L(hair_L)
    if iris_L is not None:
        vals["iris"] = _L(iris_L)

    keys = list(vals.keys())
    if len(keys) < 2:
        return 0.0, keys

    deltas: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            deltas.append(abs(vals[keys[i]] - vals[keys[j]]))

    return float(sum(deltas) / len(deltas)), keys


def park_four_season(
    undertone: str,
    avg_contrast: float,
    *,
    contrast_threshold: float = 13.0,
) -> tuple[str, bool]:
    """Map undertone + mean contrast → one of four seasons."""
    is_bright = avg_contrast >= contrast_threshold
    if undertone == "warm":
        return ("spring" if is_bright else "autumn"), is_bright
    return ("winter" if is_bright else "summer"), is_bright


def _confidence(
    undertone: str,
    skin_ab: tuple[float, float],
    avg_contrast: float,
    contrast_threshold: float,
    regions: list[str],
) -> float:
    a_star, b_star = skin_ab
    ut_margin = abs(a_star - b_star)
    ct_margin = abs(avg_contrast - contrast_threshold)

    conf = 0.38
    conf += min(0.28, ut_margin / 28.0)
    conf += min(0.22, ct_margin / 18.0)
    if len(regions) < 3:
        conf *= 0.78
    if ut_margin < 1.5:
        conf *= 0.82
    return float(min(0.88, max(0.24, conf)))


def classify_park_imcom18(
    *,
    skin_L: float,
    skin_ab: tuple[float, float],
    hair_L: float | None = None,
    iris_L: float | None = None,
    contrast_threshold: float = 13.0,
    use_cielab_l_scale: bool = True,
) -> dict:
    """Full Park IMCOM'18 seasonal analysis dict (compatible with ``extract.py``)."""
    undertone = park_undertone(skin_ab)
    avg_contrast, regions = park_mean_contrast_L(
        skin_L,
        hair_L,
        iris_L,
        use_cielab_scale=use_cielab_l_scale,
    )

    if len(regions) < 2:
        return {
            "seasonal_sixteen": "unknown",
            "seasonal_twelve": "unknown",
            "seasonal_guess": "unknown",
            "seasonal_confidence": 0.2,
            "seasonal_twelve_confidence": 0.2,
            "seasonal_method": "park_imcom18",
            "undertone_hint": undertone,
            "park_metrics": {
                "skin_a_star": round(skin_ab[0], 3),
                "skin_b_star": round(skin_ab[1], 3),
                "avg_contrast_L": round(avg_contrast, 3),
                "contrast_threshold": contrast_threshold,
                "is_bright": False,
                "regions_used": regions,
                "use_cielab_l_scale": use_cielab_l_scale,
            },
            "classifier_contributors": {},
            "seasonal_guess_top_k": [],
            "seasonal_twelve_top_k": [],
            "seasonal_notes": ["park_insufficient_regions"],
        }

    four, is_bright = park_four_season(
        undertone,
        avg_contrast,
        contrast_threshold=contrast_threshold,
    )
    twelve = FOUR_TO_TWELVE[four]
    conf = _confidence(undertone, skin_ab, avg_contrast, contrast_threshold, regions)

    alt_four = _alternate_four(undertone, is_bright)
    alt_twelve = FOUR_TO_TWELVE[alt_four]
    total = conf + conf * 0.55
    prob_main = conf / total
    prob_alt = 1.0 - prob_main

    notes: list[str] = []
    if abs(avg_contrast - contrast_threshold) < 2.5:
        notes.append("park_borderline_contrast")

    return {
        "seasonal_sixteen": "unknown",
        "seasonal_twelve": twelve,
        "seasonal_guess": four,
        "seasonal_confidence": round(conf, 4),
        "seasonal_twelve_confidence": round(conf, 4),
        "seasonal_method": "park_imcom18",
        "undertone_hint": undertone,
        "park_metrics": {
            "skin_a_star": round(skin_ab[0], 3),
            "skin_b_star": round(skin_ab[1], 3),
            "avg_contrast_L": round(avg_contrast, 3),
            "contrast_threshold": contrast_threshold,
            "is_bright": is_bright,
            "regions_used": regions,
            "use_cielab_l_scale": use_cielab_l_scale,
        },
        "classifier_contributors": {
            "park_imcom18": {
                "four": four,
                "confidence": conf,
                "undertone": undertone,
                "skin_a_star": round(skin_ab[0], 3),
                "skin_b_star": round(skin_ab[1], 3),
                "avg_contrast_L": round(avg_contrast, 3),
                "contrast_threshold": contrast_threshold,
                "is_bright": is_bright,
                "regions_used": regions,
                "use_cielab_l_scale": use_cielab_l_scale,
            },
        },
        "seasonal_guess_top_k": [
            {"season": four, "probability": round(prob_main, 4)},
            {"season": alt_four, "probability": round(prob_alt, 4)},
        ],
        "seasonal_twelve_top_k": [
            {"subtype": twelve, "probability": round(prob_main, 4)},
            {"subtype": alt_twelve, "probability": round(prob_alt, 4)},
        ],
        "seasonal_notes": notes,
    }


def _alternate_four(undertone: str, is_bright: bool) -> str:
    if undertone == "warm":
        return "autumn" if is_bright else "spring"
    return "summer" if is_bright else "winter"


def twelve_from_park_four(four: str) -> str:
    return FOUR_TO_TWELVE.get(four, "unknown")


def four_from_twelve(twelve: str) -> str | None:
    return TWELVE_TO_FOUR.get(twelve)
