from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.backends.parsing.label_maps import (
    canonical_to_parsing_result,
    celebamask_to_canonical,
    yakhyo_to_canonical,
)
from app.backends.parsing.types import ParsingResult
from app.config import settings
from app.services.yakhyo_model import resolve_face_parsing_onnx_path


def _session(onnx_path: Path) -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort.get_device() == "GPU"
        else ["CPUExecutionProvider"]
    )
    try:
        return ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception:
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def _run_bisenet(image_rgb: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    inp_size = 512
    blob = cv2.resize(image_rgb, (inp_size, inp_size), interpolation=cv2.INTER_LINEAR)
    blob = blob.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    blob = (blob - mean) / std
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: blob})[0]
    if out.ndim == 4:
        pred = np.argmax(out[0], axis=0).astype(np.uint8)
    elif out.ndim == 3:
        pred = np.argmax(out, axis=0).astype(np.uint8)
    else:
        pred = out[0].astype(np.uint8)
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def is_available() -> bool:
    if not settings.face_parsing_enabled:
        return False
    p = resolve_face_parsing_onnx_path()
    return p is not None and p.is_file()


def parse(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
) -> ParsingResult | None:
    onnx_path = resolve_face_parsing_onnx_path()
    if onnx_path is None or not onnx_path.is_file():
        return None
    try:
        sess = _session(onnx_path)
        lm = _run_bisenet(image_rgb, sess)
        if settings.face_parsing_labels == "yakhyo":
            zones = yakhyo_to_canonical(lm)
        else:
            zones = celebamask_to_canonical(lm)
        return canonical_to_parsing_result(
            zones,
            parsing_used=True,
            label_map=lm,
            parsing_backend="bisenet_resnet34",
        )
    except Exception:
        return None
