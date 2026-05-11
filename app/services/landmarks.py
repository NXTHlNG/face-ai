from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import settings
from app.services.mediapipe_model import ensure_face_landmarker_model

_landmarker = None
_landmarker_lock = threading.Lock()


@dataclass
class LandmarkResult:
    image_rgb: np.ndarray
    landmarks_px: np.ndarray
    """Shape (N, 3) x,y,z in pixels."""
    face_bbox_xywh: tuple[int, int, int, int]
    source: str = "mediapipe"


def _landmarks_to_pixels(
    lm_list: list,
    w: int,
    h: int,
) -> np.ndarray:
    arr = np.zeros((len(lm_list), 3), dtype=np.float64)
    for i, lm in enumerate(lm_list):
        arr[i, 0] = lm.x * w
        arr[i, 1] = lm.y * h
        arr[i, 2] = lm.z * w
    return arr


def _get_face_landmarker():
    global _landmarker
    import mediapipe as mp

    with _landmarker_lock:
        if _landmarker is None:
            model_path = str(ensure_face_landmarker_model())
            base = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        return _landmarker


def detect_landmarks(image_bgr: np.ndarray) -> LandmarkResult | None:
    import mediapipe as mp

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    landmarker = _get_face_landmarker()
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]
    pts = _landmarks_to_pixels(lm, w, h)

    xs = pts[:, 0]
    ys = pts[:, 1]
    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    bw = max(1, x_max - x_min)
    raw_h = max(1, y_max - y_min)
    pad = int(settings.face_bbox_forehead_pad_ratio * raw_h)
    y_min_ext = max(0, y_min - pad)
    bh = max(1, y_max - y_min_ext)

    return LandmarkResult(
        image_rgb=image_rgb,
        landmarks_px=pts,
        face_bbox_xywh=(x_min, y_min_ext, bw, bh),
        source="mediapipe",
    )


def landmark_xy(lm: np.ndarray, idx: int) -> tuple[float, float]:
    return float(lm[idx, 0]), float(lm[idx, 1])


def dist_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))
