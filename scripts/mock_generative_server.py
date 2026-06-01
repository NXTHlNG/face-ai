#!/usr/bin/env python3
"""Local mock for generative try-on (OpenAI images/edits + custom JSON)."""

from __future__ import annotations

import argparse
import base64
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

app = FastAPI(title="Mock generative try-on")


class InpaintRequest(BaseModel):
    image_b64: str
    mask_b64: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    strength: float = 0.75


def _decode_bgr_jpeg(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image_b64")
    return img


def _decode_edit_mask(b64: str, shape: tuple[int, int]) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("invalid mask_b64")
    h, w = shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _decode_openai_mask(mask_bytes: bytes, shape: tuple[int, int]) -> np.ndarray:
    arr = np.frombuffer(mask_bytes, dtype=np.uint8)
    rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise ValueError("invalid mask file")
    h, w = shape[:2]
    if rgba.shape[0] != h or rgba.shape[1] != w:
        rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_NEAREST)
    if rgba.ndim == 3 and rgba.shape[2] == 4:
        edit = rgba[:, :, 3] < 128
    else:
        edit = rgba[:, :, 0] > 127 if rgba.ndim == 3 else rgba > 127
    return (edit.astype(np.uint8) * 255)


def _tint_masked(
    image_bgr: np.ndarray,
    edit_mask: np.ndarray,
    *,
    alpha: float = 0.35,
) -> np.ndarray:
    out = image_bgr.astype(np.float32).copy()
    m = edit_mask > 127
    if not np.any(m):
        return image_bgr
    tint = np.array([40.0, 20.0, 80.0], dtype=np.float32)
    out[m] = out[m] * (1.0 - alpha) + tint * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _encode_b64_jpeg(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise ValueError("encode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.post("/v1/images/edits")
async def openai_images_edits(
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    prompt: str = Form(""),
    model: str = Form("dall-e-2"),
    n: int = Form(1),
    response_format: str = Form("b64_json"),
) -> dict:
    del prompt, model, n, response_format
    img_bytes = await image.read()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("invalid image upload")
    if mask is not None:
        mask_bytes = await mask.read()
        edit_mask = _decode_openai_mask(mask_bytes, image_bgr.shape)
        result = _tint_masked(image_bgr, edit_mask)
    else:
        edit_mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
        result = _tint_masked(image_bgr, edit_mask, alpha=0.12)
    return {"data": [{"b64_json": _encode_b64_jpeg(result)}]}


@app.post("/inpaint")
def custom_inpaint(body: InpaintRequest) -> dict:
    image_bgr = _decode_bgr_jpeg(body.image_b64)
    if body.mask_b64:
        edit_mask = _decode_edit_mask(body.mask_b64, image_bgr.shape)
        result = _tint_masked(image_bgr, edit_mask)
    else:
        edit_mask = np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255
        result = _tint_masked(image_bgr, edit_mask, alpha=0.12)
    return {"image_b64": _encode_b64_jpeg(result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
