from __future__ import annotations

import json

import cv2
import numpy as np

from app.backends.color.delta_e import delta_e_cie2000, nearest_palette_class, opencv_lab_to_cielab
from app.config import settings


def analyze_wrist_undertone(image_bgr: np.ndarray) -> dict:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    skin_mask = (
        (hue >= 3)
        & (hue <= 24)
        & (sat >= 30)
        & (sat <= 120)
        & (val >= 40)
        & (val <= 220)
    )
    vein_mask = (
        (hue >= 90)
        & (hue <= 140)
        & (sat >= 25)
        & (val >= 20)
        & (val <= 180)
    )
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    vein_u8 = (vein_mask.astype(np.uint8) * 255)
    vein_u8 = cv2.morphologyEx(vein_u8, cv2.MORPH_CLOSE, k)

    if np.count_nonzero(vein_u8 > 127) < 40:
        return {
            "undertone_hint": "neutral",
            "confidence": 0.2,
            "undertone_source": "wrist",
            "delta_e_scores": {},
        }

    L = lab[:, :, 0][vein_u8 > 127]
    a = lab[:, :, 1][vein_u8 > 127]
    b = lab[:, :, 2][vein_u8 > 127]
    vein_lab = opencv_lab_to_cielab(float(np.mean(L)), float(np.mean(a)), float(np.mean(b)))

    refs = json.loads(
        (settings.data_dir / "reference_swatches.json").read_text(encoding="utf-8")
    )["undertone_vein_lab"]
    palette = {
        "warm": tuple(refs["warm"]),
        "cool": tuple(refs["cool"]),
    }
    best, dist, scores = nearest_palette_class(vein_lab, palette)
    conf = float(max(0.35, min(0.85, 1.0 - dist / 35.0)))
    return {
        "undertone_hint": best,
        "confidence": conf,
        "undertone_source": "wrist",
        "delta_e_scores": {k: round(v, 3) for k, v in scores.items()},
    }


def fuse_undertone(
    face_undertone: str,
    wrist: dict | None,
) -> tuple[str, str]:
    if wrist is None or settings.undertone_sources == "face":
        return face_undertone, "face"
    if settings.undertone_sources == "wrist":
        return wrist.get("undertone_hint", face_undertone), "wrist"
    wu = wrist.get("undertone_hint", face_undertone)
    if face_undertone == wu:
        return face_undertone, "fused"
    if wrist.get("confidence", 0) > 0.55:
        return wu, "fused"
    return face_undertone, "fused"
