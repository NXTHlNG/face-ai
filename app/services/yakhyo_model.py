"""Загрузка ONNX face parsing из релиза yakhyo/face-parsing."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from app.config import settings


def ensure_yakhyo_onnx() -> Path:
    base = settings.models_dir
    base.mkdir(parents=True, exist_ok=True)
    path = base / settings.yakhyo_onnx_filename
    if not path.is_file():
        urllib.request.urlretrieve(str(settings.yakhyo_onnx_url), path)
    return path


def resolve_face_parsing_onnx_path() -> Path | None:
    if not settings.face_parsing_enabled:
        return None
    if settings.face_parsing_onnx_path is not None:
        return Path(settings.face_parsing_onnx_path)
    try:
        return ensure_yakhyo_onnx()
    except Exception:
        return None
