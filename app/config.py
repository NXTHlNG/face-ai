from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FACE_AI_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rules_version: str = "1.0.0"
    models_dir: Path = _PROJECT_ROOT / "models"
    landmark_backend: Literal["mediapipe", "dlib81"] = "mediapipe"
    dlib81_shape_predictor_url: str = (
        "https://github.com/codeniko/shape_predictor_81_face_landmarks/raw/master/"
        "shape_predictor_81_face_landmarks.dat"
    )
    dlib81_shape_predictor_filename: str = "shape_predictor_81_face_landmarks.dat"
    dlib_face_detector_upsample: int = 1
    mediapipe_model_url: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )
    face_parsing_enabled: bool = True
    yakhyo_onnx_url: str = (
        "https://github.com/yakhyo/face-parsing/releases/download/v0.0.2/resnet34.onnx"
    )
    yakhyo_onnx_filename: str = "yakhyo_resnet34.onnx"
    face_parsing_onnx_path: Path | None = None
    face_parsing_labels: Literal["yakhyo", "celebamask_zllrunning"] = "yakhyo"
    skip_analysis_if_photo_poor: bool = True
    face_bbox_forehead_pad_ratio: float = 0.22
    debug_save_images: bool = False
    debug_output_dir: Path = _PROJECT_ROOT / "debug_output"

    @field_validator("face_parsing_onnx_path", mode="before")
    @classmethod
    def _optional_onnx_path(cls, v):
        if v is None or v == "":
            return None
        return Path(v) if not isinstance(v, Path) else v


settings = Settings()
