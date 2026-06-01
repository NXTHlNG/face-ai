"""Map internal SegFace model names to HuggingFace checkpoint folder prefixes."""

from __future__ import annotations

# kartiknarayan/SegFace uses shortened folder names (see vendor/segface/README.md).
_HF_FOLDER_PREFIX: dict[str, str] = {
    "mobilenet": "mobilenet",
    "efficientnet": "efficientnet",
    "resnet": "resnet",
    "swin_base": "swinb",
    "swinv2_base": "swinv2b",
    "convnext_base": "convnext",
    "convnext_small": "convnext_small",
    "convnext_tiny": "convnext_tiny",
    "convnext_large": "convnext_large",
}


def local_weights_dir_name(model: str, input_size: int, dataset: str = "celeba") -> str:
    """Directory under models/segface/ (FACE_AI_SEGFACE_MODEL)."""
    return f"{model}_{dataset}_{input_size}"


def hf_weights_dir_name(model: str, input_size: int, dataset: str = "celeba") -> str:
    """Directory name on HuggingFace repo kartiknarayan/SegFace."""
    prefix = _HF_FOLDER_PREFIX.get(model, model)
    return f"{prefix}_{dataset}_{input_size}"


def hf_checkpoint_filename(model: str, input_size: int, dataset: str = "celeba") -> str:
    folder = hf_weights_dir_name(model, input_size, dataset)
    return f"{folder}/model_299.pt"
