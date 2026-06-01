from __future__ import annotations

import cv2
import numpy as np

from app.backends.color.analysis_palette import build_analysis_palette, skin_rgb_hue_trim
from app.backends.color.skin import extract_skin
from app.backends.parsing.types import ParsingResult
from app.backends.season.munsell_lookup import classify_munsell_sixteen, compute_munsell_axes
from app.backends.season.park_imcom18 import classify_park_imcom18
from app.backends.season.swatch_vote import vote_four_seasons
from app.backends.wrist.undertone import fuse_undertone
from app.config import settings
from app.pipeline.ensemble import fuse_seasonal
from app.services import color_contrast as legacy_cc


def _park_skin_ab(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    skin_ab: tuple[float, float],
) -> tuple[float, float]:
    """Use hue-trimmed skin a*/b* from the color pipeline (not raw mask mean)."""
    return skin_ab


def _run_park_classifier(
    contrast: dict,
    park_ab: tuple[float, float],
) -> dict:
    return classify_park_imcom18(
        skin_L=float(contrast["skin_L_mean"]),
        skin_ab=park_ab,
        hair_L=contrast.get("hair_L_mean"),
        iris_L=contrast.get("iris_L_mean"),
        contrast_threshold=settings.park_contrast_threshold,
        use_cielab_l_scale=settings.park_use_cielab_l_scale,
    )


def compute_contrast_and_color(
    image_rgb: np.ndarray,
    pr: ParsingResult,
    landmarks_px: np.ndarray | None = None,
    glasses_pixel_ratio: float | None = None,
    *,
    wrist_undertone: dict | None = None,
    exposure_score: float = 1.0,
) -> tuple[dict, dict, dict, dict]:
    """Returns (contrast, color_features, seasonal_analysis, analysis_palette)."""
    contrast, color = legacy_cc.compute_contrast_and_color(
        image_rgb,
        pr,
        landmarks_px,
        glasses_pixel_ratio,
    )

    skin_extra = extract_skin(image_rgb, pr.skin_mask)
    if skin_extra.get("lab"):
        color["skin_ab_mean"] = (
            round(skin_extra["ab"][0], 3),
            round(skin_extra["ab"][1], 3),
        )
        if "skin_tone_class" in skin_extra:
            color["skin_tone_class"] = skin_extra["skin_tone_class"]
            color["delta_e_scores"] = skin_extra.get("delta_e_scores")

    if exposure_score < 0.45 and settings.brightness_compensation_low_exposure > 0:
        boost = 1.0 + settings.brightness_compensation_low_exposure
        color["skin_L_compensated"] = round(
            float(contrast.get("skin_L_mean", 70)) * boost, 3
        )

    skin_ab = tuple(color.get("skin_ab_mean", (0.0, 0.0)))

    if settings.season_classifier == "park_imcom18":
        park_ab = _park_skin_ab(image_rgb, pr, skin_ab)
        seasonal = _run_park_classifier(contrast, park_ab)
        color["undertone_hint"] = seasonal["undertone_hint"]
        color["undertone_source"] = "park_skin_ab"
    else:
        undertone, ut_source = fuse_undertone(
            color.get("undertone_hint", "neutral"),
            wrist_undertone,
        )
        color["undertone_hint"] = undertone
        color["undertone_source"] = ut_source

        L_vals = [
            contrast.get("skin_L_mean"),
            contrast.get("iris_L_mean"),
            contrast.get("hair_L_mean"),
        ]
        L_valid = [x for x in L_vals if x is not None]
        L_min = min(L_valid) if L_valid else contrast["skin_L_mean"]
        L_max = max(L_valid) if L_valid else contrast["skin_L_mean"]

        axes = compute_munsell_axes(
            undertone=undertone,
            skin_ab=skin_ab,
            skin_L=float(contrast["skin_L_mean"]),
            eyes_L=contrast.get("iris_L_mean"),
            hair_L=contrast.get("hair_L_mean"),
            L_min=float(L_min),
            L_max=float(L_max),
            eyes_ab=color.get("iris_ab_mean"),
            lips_ab=color.get("lip_ab_mean"),
        )
        munsell = classify_munsell_sixteen(axes)

        skin_rgb = skin_rgb_hue_trim(image_rgb, pr.skin_mask)
        if skin_rgb is None:
            m = pr.skin_mask > 127
            if np.any(m):
                roi = image_rgb[m]
                skin_rgb = tuple(float(np.mean(roi[:, i])) for i in range(3))

        swatch = vote_four_seasons(skin_rgb)

        from app.backends.season.season_maps import SIXTEEN_TO_TWELVE, TWELVE_TO_FOUR

        if settings.season_classifier == "munsell_lookup":
            s16 = munsell["seasonal_sixteen"]
            s12 = SIXTEEN_TO_TWELVE.get(s16, s16 if s16 in TWELVE_TO_FOUR else "unknown")
            seasonal = {
                "seasonal_sixteen": s16,
                "seasonal_twelve": s12,
                "seasonal_guess": munsell["seasonal_guess"],
                "seasonal_confidence": munsell["confidence"],
                "seasonal_twelve_confidence": munsell["confidence"],
                "seasonal_method": "munsell_lookup",
                "munsell_scores": munsell["munsell_scores"],
                "classifier_contributors": {"munsell": munsell},
                "seasonal_guess_top_k": [],
                "seasonal_twelve_top_k": munsell.get("top_k_sixteen", []),
                "seasonal_notes": [],
            }
        else:
            park = _run_park_classifier(contrast, _park_skin_ab(image_rgb, pr, skin_ab))
            seasonal = fuse_seasonal(munsell, swatch, park=park, wrist_prior=wrist_undertone)
            seasonal["munsell_scores"] = munsell.get("munsell_scores", axes)

    color["seasonal_twelve"] = seasonal["seasonal_twelve"]
    color["seasonal_guess"] = seasonal["seasonal_guess"]
    color["seasonal_confidence"] = seasonal["seasonal_confidence"]
    color["seasonal_twelve_confidence"] = seasonal["seasonal_twelve_confidence"]
    color["seasonal_sixteen"] = seasonal.get("seasonal_sixteen", "unknown")

    analysis_palette = build_analysis_palette(
        image_rgb,
        pr,
        landmarks_px,
        glasses_pixel_ratio,
        seasonal_guess=seasonal.get("seasonal_guess", "unknown"),
    )

    return contrast, color, seasonal, analysis_palette
