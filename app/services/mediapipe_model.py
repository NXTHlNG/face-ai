import urllib.request
from pathlib import Path

from app.config import settings


def ensure_face_landmarker_model() -> Path:
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    path = settings.models_dir / "face_landmarker.task"
    if not path.is_file():
        urllib.request.urlretrieve(str(settings.mediapipe_model_url), path)
    return path
