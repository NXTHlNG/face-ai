from __future__ import annotations

import logging

import numpy as np

from app.backends.parsing.label_maps import (
    canonical_to_parsing_result,
    farl_label_map_to_canonical,
)
from app.backends.parsing.types import ParsingResult
from app.config import settings

_log = logging.getLogger(__name__)

_face_detector = None
_face_parser = None
_last_error: str | None = None


def last_error() -> str | None:
    return _last_error


def is_available() -> bool:
    global _last_error
    if not settings.face_parsing_enabled:
        _last_error = "face_parsing_disabled"
        return False
    try:
        import torch  # noqa: F401
        import facer  # noqa: F401

        _last_error = None
        return True
    except ImportError as e:
        _last_error = str(e)
        return False


def _landmark_bbox_xyxy(landmarks_px: np.ndarray) -> np.ndarray | None:
    if landmarks_px is None or len(landmarks_px) == 0:
        return None
    xs = landmarks_px[:, 0]
    ys = landmarks_px[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _primary_face_index(faces: dict, landmarks_px: np.ndarray) -> int:
    rects = faces.get("rects")
    if rects is None:
        return 0
    if hasattr(rects, "detach"):
        rects = rects.detach().cpu().numpy()
    rects = np.asarray(rects, dtype=np.float32)
    if rects.ndim == 1:
        return 0
    if len(rects) <= 1:
        return 0

    lm = np.asarray(landmarks_px, dtype=np.float32)
    if lm.ndim == 2 and lm.shape[1] >= 2:
        lm_xy = lm[:, :2]
    else:
        lm_xy = lm.reshape(-1, 2)
    lm_center = np.mean(lm_xy, axis=0)
    lm_box = _landmark_bbox_xyxy(lm_xy)

    scores = faces.get("scores")
    if scores is not None and hasattr(scores, "detach"):
        scores = scores.detach().cpu().numpy()
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32)

    def _contains_center(rect: np.ndarray) -> bool:
        return (
            rect[0] <= lm_center[0] <= rect[2]
            and rect[1] <= lm_center[1] <= rect[3]
        )

    candidates = [i for i, rect in enumerate(rects) if _contains_center(rect)]
    if not candidates and lm_box is not None:
        ious = [_bbox_iou(lm_box, rect) for rect in rects]
        if max(ious) > 0.05:
            return int(np.argmax(ious))
        candidates = list(range(len(rects)))
    elif not candidates:
        candidates = list(range(len(rects)))

    if scores is not None and len(scores) == len(rects):
        return max(candidates, key=lambda i: scores[i])

    if lm_box is not None:
        return max(candidates, key=lambda i: _bbox_iou(lm_box, rects[i]))

    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    return max(candidates, key=lambda i: areas[i])


def _ensure_models():
    global _face_detector, _face_parser, _last_error
    if _face_detector is not None and _face_parser is not None:
        return _face_detector, _face_parser
    import torch
    import facer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _face_detector = facer.face_detector("retinaface/mobilenet", device=device)
    _face_parser = facer.face_parser(settings.farl_model_name, device=device)
    _last_error = None
    return _face_detector, _face_parser


def parse(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
) -> ParsingResult | None:
    global _last_error
    if not is_available():
        return None
    try:
        import torch
        import facer

        detector, parser = _ensure_models()
        device = next(parser.parameters()).device

        # facer expects BCHW float 0..1; helper handles RGB HWC uint8
        image_bchw = facer.hwc2bchw(
            torch.from_numpy(image_rgb.copy()).to(device=device)
        )

        with torch.inference_mode():
            faces = detector(image_bchw)
            if not faces or faces.get("rects") is None or len(faces["rects"]) == 0:
                _last_error = "no_face_detected_by_retinaface"
                return None
            faces = parser(image_bchw, faces)

        seg = faces.get("seg") if isinstance(faces, dict) else None
        if seg is None:
            _last_error = "missing_seg_in_facer_output"
            return None

        logits = seg.get("logits")
        if logits is None:
            _last_error = "missing_logits_in_facer_seg"
            return None

        face_idx = _primary_face_index(faces, landmarks_px)
        if logits.dim() == 4:
            logits_np = logits[face_idx].detach().cpu().numpy()
        else:
            logits_np = logits.detach().cpu().numpy()

        label_map = np.argmax(logits_np, axis=0).astype(np.uint8)
        label_names = seg.get("label_names")
        zones = farl_label_map_to_canonical(
            label_map,
            settings.farl_model_name,
            label_names=label_names,
        )
        h, w = image_rgb.shape[:2]
        import cv2

        for k in zones:
            if zones[k].shape[:2] != (h, w):
                zones[k] = cv2.resize(zones[k], (w, h), interpolation=cv2.INTER_NEAREST)

        _last_error = None
        return canonical_to_parsing_result(
            zones,
            parsing_used=True,
            label_map=label_map,
            parsing_backend="farl_b",
        )
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
        _log.warning("FaRL-B parsing failed: %s", _last_error)
        return None
