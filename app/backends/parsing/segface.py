from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from app.backends.parsing.label_maps import canonical_to_parsing_result, segface_to_canonical
from app.segface_weights import hf_weights_dir_name, local_weights_dir_name
from app.backends.parsing.types import ParsingResult
from app.config import settings

_log = logging.getLogger(__name__)

_VENDOR_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "segface"
_model = None
_last_error: str | None = None


def last_error() -> str | None:
    return _last_error


def is_available() -> bool:
    global _last_error
    if not settings.face_parsing_enabled:
        _last_error = "face_parsing_disabled"
        return False
    if not _VENDOR_ROOT.is_dir():
        _last_error = f"vendor segface missing: {_VENDOR_ROOT}"
        return False
    try:
        import torch  # noqa: F401

        _last_error = None
        return True
    except ImportError as e:
        _last_error = str(e)
        return False


def _weights_dir_name() -> str:
    return local_weights_dir_name(
        settings.segface_model, settings.segface_input_size
    )


def _resolve_weights() -> Path | None:
    if settings.segface_weights_path and settings.segface_weights_path.is_file():
        return settings.segface_weights_path
    root = settings.models_dir / "segface"
    candidates = [
        root
        / local_weights_dir_name(settings.segface_model, settings.segface_input_size)
        / "model_299.pt",
        root
        / hf_weights_dir_name(settings.segface_model, settings.segface_input_size)
        / "model_299.pt",
    ]
    for named in candidates:
        if named.is_file():
            return named
    legacy = settings.models_dir / "segface.pth"
    return legacy if legacy.is_file() else None


def _ensure_vendor_path() -> None:
    root = str(_VENDOR_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_model():
    global _model, _last_error
    if _model is not None:
        return _model

    import torch
    import torch.nn.functional as F

    weights = _resolve_weights()
    if weights is None:
        raise FileNotFoundError(
            "SegFace weights not found. Run: python scripts/download_segface_weights.py "
            f"or set FACE_AI_SEGFACE_WEIGHTS_PATH (expected "
            f"models/segface/{_weights_dir_name()}/model_299.pt)"
        )

    _ensure_vendor_path()
    from network import get_model  # type: ignore[import-untyped]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        settings.segface_backbone,
        settings.segface_input_size,
        settings.segface_model,
    )
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)

    state = checkpoint.get("state_dict_backbone", checkpoint)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)

    _model = (model, device, F)
    _last_error = None
    return _model


def parse(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
) -> ParsingResult | None:
    del landmarks_px  # full-frame seg; landmarks used upstream for bbox only
    global _last_error
    if not is_available():
        return None
    weights = _resolve_weights()
    if weights is None:
        _last_error = "segface_weights_missing"
        return None
    try:
        import torch

        model, device, F = _ensure_model()
        h, w = image_rgb.shape[:2]
        sz = settings.segface_input_size
        blob = cv2.resize(image_rgb, (sz, sz), interpolation=cv2.INTER_LINEAR)
        t = torch.from_numpy(blob).permute(2, 0, 1).float().to(device) / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        t = ((t - mean) / std).unsqueeze(0)

        dummy_labels = {"segmentation": torch.zeros(1, sz, sz, device=device)}
        dummy_dataset = torch.zeros(1, dtype=torch.long, device=device)

        with torch.no_grad():
            seg_output = model(t, dummy_labels, dummy_dataset)
            mask = F.interpolate(
                seg_output,
                size=(sz, sz),
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(mask.softmax(dim=1), dim=1)[0].cpu().numpy().astype(np.uint8)

        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        zones = segface_to_canonical(pred)
        _last_error = None
        return canonical_to_parsing_result(
            zones,
            parsing_used=True,
            label_map=pred,
            parsing_backend="segface",
        )
    except Exception as exc:
        _last_error = str(exc)
        _log.warning("SegFace parse failed: %s", exc, exc_info=True)
        return None
