from __future__ import annotations

import base64

import cv2
import numpy as np

from app.backends.parsing.types import ParsingResult


def _mask_overlay_b64(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> str:
    vis = image_rgb.copy()
    m = mask > 127
    vis[m] = (
        vis[m].astype(np.float32) * 0.45 + np.array(color, dtype=np.float32) * 0.55
    ).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def combined_features_b64(image_rgb: np.ndarray, pr: ParsingResult) -> str:
    h, w = image_rgb.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    m = pr.skin_mask > 127
    out[m] = image_rgb[m]
    if pr.hair_mask is not None:
        hm = pr.hair_mask > 127
        out[hm] = (out[hm].astype(np.float32) * 0.6 + np.array([80, 60, 40]) * 0.4).astype(np.uint8)
    if pr.eye_region_mask is not None:
        em = pr.eye_region_mask > 127
        out[em] = image_rgb[em]
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_mask_preview(image_rgb: np.ndarray, pr: ParsingResult) -> dict:
    return {
        "skin_mask_b64": _mask_overlay_b64(image_rgb, pr.skin_mask, (255, 180, 140)),
        "combined_features_b64": combined_features_b64(image_rgb, pr),
    }
