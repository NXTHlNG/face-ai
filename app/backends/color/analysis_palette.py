"""RGB swatches from face regions used in seasonal color analysis (Park-style personal palette)."""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from app.backends.color.aua_heuristics import (
    brow_lab_from_mask,
    hair_lab_luminance_trim,
    skin_lab_hue_trim,
    skin_trim_pixels,
)
from app.backends.color.lab_utils import ab_from_opencv_lab, mean_lab, median_lab
from app.config import settings

_SEASON_ANCHORS: dict[str, list[list[int]]] | None = None


def _load_season_anchors() -> dict[str, list[list[int]]]:
    global _SEASON_ANCHORS
    if _SEASON_ANCHORS is None:
        raw = json.loads(
            (settings.data_dir / "reference_swatches.json").read_text(encoding="utf-8")
        )
        _SEASON_ANCHORS = raw["season_rgb_anchors"]
    return _SEASON_ANCHORS


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def opencv_lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    px = np.array([[[lab[0], lab[1], lab[2]]]], dtype=np.uint8)
    rgb = cv2.cvtColor(px, cv2.COLOR_LAB2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _luminance_rgb_row(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, 0].astype(np.float32)
    g = rgb[:, 1].astype(np.float32)
    b = rgb[:, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def skin_rgb_hue_trim(
    image_rgb: np.ndarray,
    skin_mask: np.ndarray,
    *,
    min_pixels: int = 24,
) -> tuple[int, int, int] | None:
    """Median RGB on hue-trimmed skin — same pixel set as ``skin_lab_hue_trim``."""
    trimmed = skin_trim_pixels(image_rgb, skin_mask, min_pixels=min_pixels)
    if trimmed is None:
        lab = median_lab(image_rgb, skin_mask) or mean_lab(image_rgb, skin_mask)
        return opencv_lab_to_rgb(lab) if lab else None
    if trimmed.shape[0] < 8:
        lab = median_lab(image_rgb, skin_mask) or mean_lab(image_rgb, skin_mask)
        return opencv_lab_to_rgb(lab) if lab else None
    return tuple(int(np.median(trimmed[:, i])) for i in range(3))  # type: ignore[return-value]


def swatch_from_lab(region: str, lab: tuple[float, float, float]) -> dict[str, Any]:
    rgb = opencv_lab_to_rgb(lab)
    ab = ab_from_opencv_lab(lab)
    return {
        "region": region,
        "rgb": list(rgb),
        "hex": rgb_to_hex(rgb),
        "lab_opencv": [round(lab[0], 3), round(lab[1], 3), round(lab[2], 3)],
        "lab_cielab": [round(lab[0] * 100.0 / 255.0, 3), round(ab[0], 3), round(ab[1], 3)],
    }


def swatch_from_rgb(region: str, rgb: tuple[int, int, int]) -> dict[str, Any]:
    lab = cv2.cvtColor(
        np.array([[list(rgb)]], dtype=np.uint8),
        cv2.COLOR_RGB2LAB,
    )[0, 0]
    lab_t = (float(lab[0]), float(lab[1]), float(lab[2]))
    ab = ab_from_opencv_lab(lab_t)
    return {
        "region": region,
        "rgb": list(rgb),
        "hex": rgb_to_hex(rgb),
        "lab_opencv": [round(lab_t[0], 3), round(lab_t[1], 3), round(lab_t[2], 3)],
        "lab_cielab": [round(lab_t[0] * 100.0 / 255.0, 3), round(ab[0], 3), round(ab[1], 3)],
    }


def build_analysis_palette(
    image_rgb: np.ndarray,
    pr: Any,
    landmarks_px: np.ndarray | None,
    glasses_pixel_ratio: float | None,
    *,
    seasonal_guess: str = "unknown",
) -> dict[str, Any]:
    """Face-region swatches + optional season reference anchors used in swatch voting."""
    from app.services.color_contrast import _mean_lab, _resolve_iris_lab, _resolve_lip_lab

    face: list[dict[str, Any]] = []

    skin_lab_opencv = skin_lab_hue_trim(image_rgb, pr.skin_mask) or _mean_lab(
        image_rgb, pr.skin_mask
    )

    skin_rgb = skin_rgb_hue_trim(image_rgb, pr.skin_mask)
    if skin_rgb is not None:
        face.append(swatch_from_rgb("skin", skin_rgb))
    elif skin_lab_opencv:
        face.append(swatch_from_lab("skin", skin_lab_opencv))

    hair_lab = hair_lab_luminance_trim(image_rgb, pr.hair_mask, pr.skin_mask)
    if hair_lab is None:
        hair_lab = _mean_lab(image_rgb, pr.hair_mask)
    if hair_lab:
        face.append(swatch_from_lab("hair", hair_lab))

    brow_lab = brow_lab_from_mask(image_rgb, pr.brow_mask, pr.hair_mask)
    if brow_lab is None:
        brow_lab = _mean_lab(image_rgb, pr.brow_mask)
    if brow_lab:
        face.append(swatch_from_lab("brow", brow_lab))

    hair_L = hair_lab[0] if hair_lab else None
    iris_lab = _resolve_iris_lab(
        image_rgb, pr, landmarks_px, glasses_pixel_ratio
    )
    if iris_lab:
        face.append(swatch_from_lab("iris", iris_lab))

    lip_lab = _resolve_lip_lab(image_rgb, pr, landmarks_px)
    if lip_lab:
        face.append(swatch_from_lab("lip", lip_lab))

    season_reference: list[dict[str, Any]] | None = None
    if seasonal_guess in ("spring", "summer", "autumn", "winter"):
        anchors = _load_season_anchors()[seasonal_guess]
        season_reference = [
            swatch_from_rgb(f"{seasonal_guess}_ref_{idx}", (rgb[0], rgb[1], rgb[2]))
            for idx, rgb in enumerate(anchors, start=1)
        ]

    return {
        "face": face,
        "season_reference": season_reference,
    }
