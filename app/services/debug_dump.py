"""Сохранение промежуточных изображений в debug-папку (включить FACE_AI_DEBUG_SAVE_IMAGES)."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.color_contrast import build_iris_ring_union_debug_mask
from app.services.face_parsing import ParsingResult
from app.services.landmarks import LandmarkResult


def new_run_dir() -> Path | None:
    if not settings.debug_save_images:
        return None
    base = settings.debug_output_dir
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:8]
    run = base / f"{stamp}_{uid}"
    run.mkdir(parents=True, exist_ok=True)
    return run


def _write_text(run_dir: Path | None, filename: str, text: str) -> None:
    if run_dir is None:
        return
    (run_dir / filename).write_text(text, encoding="utf-8")


def save_input(run_dir: Path | None, image_bgr: np.ndarray) -> None:
    if run_dir is None:
        return
    cv2.imwrite(str(run_dir / "01_input.jpg"), image_bgr)


def save_landmarks(run_dir: Path | None, image_bgr: np.ndarray, lr: LandmarkResult) -> None:
    if run_dir is None:
        return
    vis = image_bgr.copy()
    lm = lr.landmarks_px
    for i in range(len(lm)):
        x, y = int(lm[i, 0]), int(lm[i, 1])
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    cv2.rectangle(
        vis,
        (lr.face_bbox_xywh[0], lr.face_bbox_xywh[1]),
        (
            lr.face_bbox_xywh[0] + lr.face_bbox_xywh[2],
            lr.face_bbox_xywh[1] + lr.face_bbox_xywh[3],
        ),
        (0, 165, 255),
        1,
    )
    cv2.imwrite(str(run_dir / "02_landmarks.jpg"), vis)


def save_face_contour_debug(
    run_dir: Path | None,
    image_bgr: np.ndarray,
    contour_px: np.ndarray | None,
) -> None:
    if run_dir is None or contour_px is None or len(contour_px) < 3:
        return
    vis = image_bgr.copy()
    arr = np.asarray(contour_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [arr], True, (0, 200, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(run_dir / "07_face_shape_contour.jpg"), vis)


def save_label_map_raw(run_dir: Path | None, label_map: np.ndarray | None) -> None:
    if run_dir is None or label_map is None:
        return
    mx = float(label_map.max()) if label_map.size else 1.0
    scaled = (label_map.astype(np.float32) / max(mx, 1.0) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    cv2.imwrite(str(run_dir / "03_label_map_color.jpg"), colored)


def save_masks(run_dir: Path | None, pr: ParsingResult) -> None:
    if run_dir is None:
        return
    cv2.imwrite(str(run_dir / "04_mask_skin.png"), pr.skin_mask)
    cv2.imwrite(str(run_dir / "04_mask_hair.png"), pr.hair_mask)
    cv2.imwrite(str(run_dir / "04_mask_brows.png"), pr.brow_mask)
    if pr.lip_mask is not None:
        cv2.imwrite(str(run_dir / "04_mask_lips.png"), pr.lip_mask)
    if pr.eye_region_mask is not None:
        cv2.imwrite(str(run_dir / "04_mask_eye_region.png"), pr.eye_region_mask)
    if pr.eye_glass_mask is not None:
        cv2.imwrite(str(run_dir / "04_mask_glasses.png"), pr.eye_glass_mask)


def save_iris_ring_debug(
    run_dir: Path | None,
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray | None,
    pr: ParsingResult,
    glasses_pixel_ratio: float | None,
) -> None:
    if run_dir is None:
        return
    mask = build_iris_ring_union_debug_mask(
        image_rgb,
        landmarks_px,
        pr,
        glasses_pixel_ratio,
    )
    if mask is None:
        _write_text(run_dir, "06_iris_ring.txt", "iris ring mask unavailable\n")
        return
    cv2.imwrite(str(run_dir / "06_iris_ring_mask.png"), mask)
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy().astype(np.float32)
    cyan = np.array([255.0, 255.0, 0.0], dtype=np.float32)
    sel = mask > 127
    overlay[sel] = overlay[sel] * 0.52 + cyan * 0.48
    cv2.imwrite(str(run_dir / "06_iris_ring_overlay.jpg"), np.clip(overlay, 0, 255).astype(np.uint8))


def save_parsing_overlay(run_dir: Path | None, image_rgb: np.ndarray, pr: ParsingResult) -> None:
    if run_dir is None:
        return
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = 0.38
    layers = [
        (pr.skin_mask, (180, 200, 220)),
        (pr.hair_mask, (80, 120, 200)),
        (pr.brow_mask, (40, 40, 40)),
    ]
    if pr.eye_glass_mask is not None:
        layers.append((pr.eye_glass_mask, (0, 255, 255)))
    for mask, color in layers:
        m = mask > 127
        for c in range(3):
            ch = bgr[:, :, c]
            ch[m] = ch[m] * (1 - alpha) + color[c] * alpha
    out = np.clip(bgr, 0, 255).astype(np.uint8)
    cv2.imwrite(str(run_dir / "05_parsing_overlay.jpg"), out)


def save_meta(
    run_dir: Path | None,
    *,
    parsing_used: bool,
    passes_gate: bool,
    issues: list[str],
    landmark_backend: str | None = None,
) -> None:
    if run_dir is None:
        return
    lines = [
        f"parsing_used_yakhyo_or_onnx={parsing_used}",
        f"photo_passes_gate={passes_gate}",
        f"issues={issues}",
    ]
    if landmark_backend:
        lines.append(f"landmark_backend={landmark_backend}")
    _write_text(run_dir, "00_meta.txt", "\n".join(lines) + "\n")


def save_gate_blocked(run_dir: Path | None, issues: list[str]) -> None:
    _write_text(run_dir, "02_gate_blocked.txt", "Gate blocked. Issues:\n" + "\n".join(issues) + "\n")


def save_no_face(run_dir: Path | None) -> None:
    _write_text(run_dir, "02_no_face.txt", "Face not detected.\n")
