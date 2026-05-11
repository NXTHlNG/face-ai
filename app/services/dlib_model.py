"""Путь к shape_predictor_81_face_landmarks.dat — vendor-клон или models/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from app.config import settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_VENDOR_DAT = (
    _PROJECT_ROOT
    / "vendor"
    / "shape_predictor_81_face_landmarks"
    / "shape_predictor_81_face_landmarks.dat"
)


def resolve_dlib81_dat_path() -> Path:
    if _VENDOR_DAT.is_file():
        return _VENDOR_DAT
    models_path = settings.models_dir / settings.dlib81_shape_predictor_filename
    if models_path.is_file():
        return models_path
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(str(settings.dlib81_shape_predictor_url), models_path)
    return models_path
