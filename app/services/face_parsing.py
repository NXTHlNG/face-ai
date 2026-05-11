from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from app.config import settings
from app.services.yakhyo_model import resolve_face_parsing_onnx_path

FACE_OVAL_IDX = np.array(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    dtype=np.int32,
)

LEFT_BROW_IDX = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_BROW_IDX = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


@dataclass
class ParsingResult:
    skin_mask: np.ndarray
    hair_mask: np.ndarray
    brow_mask: np.ndarray
    eye_glass_mask: np.ndarray | None
    lip_mask: np.ndarray | None
    eye_region_mask: np.ndarray | None
    parsing_used: bool
    label_map: np.ndarray | None


def _session(onnx_path: Path) -> ort.InferenceSession:
    if ort.get_device() == "GPU":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
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


def _fill_convex_poly_int32(
    mask: np.ndarray,
    pts: list[list[float]],
) -> None:
    if len(pts) < 3:
        return
    h, w = mask.shape[:2]
    arr = np.asarray(pts, dtype=np.float32)
    hull = cv2.convexHull(arr)
    poly = np.round(hull).astype(np.int32).reshape(-1, 2)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    if poly.shape[0] < 3:
        return
    cv2.fillConvexPoly(mask, poly, 255)


def labels_to_masks(
    label_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if settings.face_parsing_labels == "yakhyo":
        return _labels_to_masks_yakhyo(label_map)
    return _labels_to_masks_celebamask_zllrunning(label_map)


def _labels_to_masks_yakhyo(label_map: np.ndarray) -> tuple[np.ndarray, ...]:
    """Индексы как в yakhyo/face-parsing utils/prepare_labels.py (enumerate(attributes, 1))."""
    skin = (label_map == 1).astype(np.uint8) * 255
    hair = (label_map == 17).astype(np.uint8) * 255
    brow = ((label_map == 2) | (label_map == 3)).astype(np.uint8) * 255
    glasses = (label_map == 6).astype(np.uint8) * 255
    lip = ((label_map == 12) | (label_map == 13)).astype(np.uint8) * 255
    eye_region = ((label_map == 4) | (label_map == 5)).astype(np.uint8) * 255
    return skin, hair, brow, glasses, lip, eye_region


def _labels_to_masks_celebamask_zllrunning(label_map: np.ndarray) -> tuple[np.ndarray, ...]:
    """Классическая нумерация face-parsing.PyTorch / части CelebAMask-HQ."""
    skin = (label_map == 1).astype(np.uint8) * 255
    hair = (label_map == 13).astype(np.uint8) * 255
    brow = ((label_map == 6) | (label_map == 7)).astype(np.uint8) * 255
    glasses = (label_map == 3).astype(np.uint8) * 255
    lip = ((label_map == 11) | (label_map == 12)).astype(np.uint8) * 255
    eye_region = ((label_map == 4) | (label_map == 5)).astype(np.uint8) * 255
    return skin, hair, brow, glasses, lip, eye_region


DLIB_LEFT_BROW = list(range(17, 22))
DLIB_RIGHT_BROW = list(range(22, 27))


def fallback_masks_dlib81(
    image_shape: tuple[int, int],
    landmarks_px: np.ndarray,
) -> ParsingResult:
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_all = [[float(landmarks_px[i, 0]), float(landmarks_px[i, 1])] for i in range(len(landmarks_px))]
    _fill_convex_poly_int32(mask, pts_all)
    brow = np.zeros_like(mask)
    for idx_list in (DLIB_LEFT_BROW, DLIB_RIGHT_BROW):
        bpts = [[float(landmarks_px[i, 0]), float(landmarks_px[i, 1])] for i in idx_list]
        _fill_convex_poly_int32(brow, bpts)
    hair = np.zeros_like(mask)
    top_y = int(np.clip(np.min(landmarks_px[:, 1]) - (h * 0.08), 0, h - 1))
    cv2.rectangle(hair, (0, 0), (w - 1, top_y + max(1, h // 25)), 255, thickness=-1)
    return ParsingResult(
        skin_mask=mask,
        hair_mask=hair,
        brow_mask=brow,
        eye_glass_mask=None,
        lip_mask=None,
        eye_region_mask=None,
        parsing_used=False,
        label_map=None,
    )


def fallback_masks_from_landmarks(
    image_shape: tuple[int, int],
    landmarks_px: np.ndarray,
) -> ParsingResult:
    if landmarks_px.shape[0] == 81:
        return fallback_masks_dlib81(image_shape, landmarks_px)
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = []
    for i in FACE_OVAL_IDX:
        if i < len(landmarks_px):
            pts.append([landmarks_px[i, 0], landmarks_px[i, 1]])
    _fill_convex_poly_int32(mask, pts)
    brow = np.zeros_like(mask)
    for idx_list in (LEFT_BROW_IDX, RIGHT_BROW_IDX):
        bpts = []
        for i in idx_list:
            if i < len(landmarks_px):
                bpts.append([landmarks_px[i, 0], landmarks_px[i, 1]])
        _fill_convex_poly_int32(brow, bpts)
    hair = np.zeros_like(mask)
    top_y = int(np.clip(np.min(landmarks_px[:, 1]) - (h * 0.08), 0, h - 1))
    cv2.rectangle(hair, (0, 0), (w - 1, top_y + max(1, h // 25)), 255, thickness=-1)
    return ParsingResult(
        skin_mask=mask,
        hair_mask=hair,
        brow_mask=brow,
        eye_glass_mask=None,
        lip_mask=None,
        eye_region_mask=None,
        parsing_used=False,
        label_map=None,
    )


def parse_face(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
) -> ParsingResult:
    onnx_path = resolve_face_parsing_onnx_path()
    if onnx_path is not None and onnx_path.is_file():
        try:
            sess = _session(onnx_path)
            lm = _run_bisenet(image_rgb, sess)
            skin, hair, brow, glasses, lip_m, eye_m = labels_to_masks(lm)
            return ParsingResult(
                skin_mask=skin,
                hair_mask=hair,
                brow_mask=brow,
                eye_glass_mask=glasses,
                lip_mask=lip_m,
                eye_region_mask=eye_m,
                parsing_used=True,
                label_map=lm,
            )
        except Exception:
            pass
    return fallback_masks_from_landmarks(image_rgb.shape, landmarks_px)


def glasses_pixel_ratio(pr: ParsingResult) -> float | None:
    if pr.eye_glass_mask is None:
        return None
    return float(np.mean(pr.eye_glass_mask > 127))
