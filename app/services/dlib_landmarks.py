"""81 точка iBUG+codeniko через dlib shape predictor + frontal face detector."""

from __future__ import annotations

import threading

import cv2
import numpy as np

from app.config import settings
from app.exceptions import DlibNotInstalledError
from app.services.dlib_model import resolve_dlib81_dat_path
from app.services.landmarks import LandmarkResult

DLIB_HINT_RU = (
    "Для режима dlib81 нужен пакет dlib. На Windows без компилятора проще всего: "
    "`conda install -c conda-forge dlib` в окружении с вашим Python. "
    "Либо установите CMake + Visual Studio Build Tools (C++) и выполните "
    "`pip install -r requirements-dlib.txt`. См. README."
)

_detector = None
_predictor = None
_lock = threading.Lock()


def _get_detector_and_predictor():
    global _detector, _predictor
    try:
        import dlib
    except ImportError as e:
        raise DlibNotInstalledError(DLIB_HINT_RU) from e

    with _lock:
        if _predictor is None:
            _detector = dlib.get_frontal_face_detector()
            dat_path = resolve_dlib81_dat_path()
            _predictor = dlib.shape_predictor(str(dat_path))
        return _detector, _predictor


def detect_landmarks_dlib81(image_bgr: np.ndarray) -> LandmarkResult | None:
    detector, predictor = _get_detector_and_predictor()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, settings.dlib_face_detector_upsample)
    if len(faces) == 0:
        return None

    rect = max(faces, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)
    n = shape.num_parts
    pts = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        p = shape.part(i)
        pts[i, 0] = float(p.x)
        pts[i, 1] = float(p.y)

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    left, top, right, bottom = rect.left(), rect.top(), rect.right(), rect.bottom()
    x_min, y_min = int(left), int(top)
    bw = max(1, int(right - left))
    raw_h = max(1, int(bottom - top))
    pad = int(settings.face_bbox_forehead_pad_ratio * raw_h)
    y_min_ext = max(0, y_min - pad)
    bh = max(1, int(bottom - y_min_ext))

    return LandmarkResult(
        image_rgb=image_rgb,
        landmarks_px=pts,
        face_bbox_xywh=(x_min, y_min_ext, bw, bh),
        source="dlib81",
    )
