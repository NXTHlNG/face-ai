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

    rules_version: str = "2.0.0"
    models_dir: Path = _PROJECT_ROOT / "models"
    data_dir: Path = _PROJECT_ROOT / "app" / "data"

    # Landmarks / detection
    landmark_backend: Literal["mediapipe", "dlib81"] = "mediapipe"
    detector_backend: Literal["mediapipe", "retinaface"] = "mediapipe"

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

    # Face parsing
    face_parsing_enabled: bool = True
    parsing_backend: Literal[
        "bisenet_resnet34",
        "farl_b",
        "segface",
        "landmark_fallback",
    ] = "bisenet_resnet34"
    parsing_fallback_chain: str = "bisenet_resnet34,farl_b,landmark_fallback"

    yakhyo_onnx_url: str = (
        "https://github.com/yakhyo/face-parsing/releases/download/v0.0.2/resnet34.onnx"
    )
    yakhyo_onnx_filename: str = "yakhyo_resnet34.onnx"
    face_parsing_onnx_path: Path | None = None
    face_parsing_labels: Literal["yakhyo", "celebamask_zllrunning"] = "yakhyo"

    farl_model_name: str = "farl/celebm/448"
    farl_skin_threshold: float = 0.5
    segface_weights_path: Path | None = None
    segface_input_size: int = 512
    segface_backbone: str = "segface_celeb"
    segface_model: str = "mobilenet"

    # Color extraction backends
    skin_color_backend: Literal["mean_lab", "xmeans_hsv_deltae"] = "mean_lab"
    hair_color_backend: Literal["mean_lab", "kmeans_lab_k3"] = "mean_lab"
    iris_color_backend: Literal["luminance_ring", "dlib_circular"] = "luminance_ring"
    lip_color_backend: Literal["parsing_mask", "brightness_clusters"] = "brightness_clusters"

    # Season classification
    season_classifier: Literal["munsell_lookup", "ensemble", "park_imcom18"] = "ensemble"
    undertone_sources: Literal["face", "wrist", "fused"] = "face"
    # Park et al. IMCOM'18: mean pairwise |ΔL| threshold (CIELAB L 0–100, §3.2.4)
    park_contrast_threshold: float = 13.0
    park_use_cielab_l_scale: bool = True

    # LLM analysis (primary semantics; rules fallback when disabled/unavailable)
    llm_enabled: bool = True
    llm_api_url: str = ""
    llm_api_key: str = ""

    # Generative try-on Model API (HTTP); "none" = disabled
    generative_api_url: str = "none"
    generative_api_key: str = ""
    generative_transport: Literal["openai_images_edit", "custom_json", "gemini_native"] = (
        "openai_images_edit"
    )
    generative_model: str = "dall-e-2"
    generative_timeout_s: float = 90.0
    generative_strength: float = 0.75
    generative_use_mask: bool = True
    # Blend model output with original using parsing mask (critical for makeup preservation)
    generative_composite_mask: bool = False
    # OpenAI images/edits size: auto preserves input aspect ratio (gpt-image-*)
    generative_image_size: str = "auto"
    # auto | b64_json | omit — gpt-image-* and many proxies reject response_format
    generative_response_format: str = "auto"

    # Try-on / outfit
    tryon_default_categories: str = "makeup,glasses,hairstyle"
    outfit_inline_min_ratio: float = 0.15
    products_file: Path | None = None
    makeup_db_file: Path | None = None

    # Photo / debug
    skip_analysis_if_photo_poor: bool = True
    face_bbox_forehead_pad_ratio: float = 0.22
    debug_save_images: bool = False
    debug_output_dir: Path = _PROJECT_ROOT / "debug_output"
    mask_preview_enabled: bool = True

    # Post-process
    skin_blur_sigma_ratio: float = 0.012
    gamma_correction: float = 1.0
    brightness_compensation_low_exposure: float = 0.2

    @field_validator(
        "face_parsing_onnx_path",
        "segface_weights_path",
        "products_file",
        "makeup_db_file",
        mode="before",
    )
    @classmethod
    def _optional_path(cls, v):
        if v is None or v == "":
            return None
        return Path(v) if not isinstance(v, Path) else v

    def resolved_products_path(self) -> Path:
        if self.products_file is not None:
            return self.products_file
        return self.data_dir / "products.json"

    def resolved_makeup_db_path(self) -> Path:
        if self.makeup_db_file is not None:
            return self.makeup_db_file
        return self.data_dir / "makeup_db.json"

    @property
    def llm_available(self) -> bool:
        return self.llm_enabled and bool(self.llm_api_url.strip())

    @property
    def resolved_generative_api_key(self) -> str:
        """Generative key, else LLM key (same provider / OpenAI-compatible gateway)."""
        key = self.generative_api_key.strip()
        if key:
            return key
        return self.llm_api_key.strip()

    @property
    def generative_available(self) -> bool:
        url = self.generative_api_url.strip().lower()
        return bool(url) and url != "none"


settings = Settings()
