from __future__ import annotations

import base64
import json
import logging
import uuid
from typing import Any, Callable
from urllib import error, request

import cv2
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

_GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"


def _bgr_to_png_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("failed to encode image as PNG")
    return buf.tobytes()


def _mask_to_openai_rgba(edit_mask: np.ndarray, shape: tuple[int, int]) -> bytes:
    h, w = shape[:2]
    if edit_mask.shape[:2] != (h, w):
        edit_mask = cv2.resize(edit_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    edit = edit_mask > 127
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(edit, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise ValueError("failed to encode mask as PNG")
    return buf.tobytes()


def _mask_to_png_bytes(edit_mask: np.ndarray, shape: tuple[int, int]) -> bytes:
    h, w = shape[:2]
    if edit_mask.shape[:2] != (h, w):
        edit_mask = cv2.resize(edit_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".png", edit_mask.astype(np.uint8))
    if not ok:
        raise ValueError("failed to encode mask as PNG")
    return buf.tobytes()


def _decode_image_b64(image_b64: str) -> np.ndarray:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode result image")
    return img


def _align_to_source_size(image_bgr: np.ndarray, source_shape: tuple[int, ...]) -> np.ndarray:
    target_h, target_w = source_shape[:2]
    h, w = image_bgr.shape[:2]
    if h == target_h and w == target_w:
        return image_bgr
    logger.info("generative output resized %sx%s -> %sx%s", w, h, target_w, target_h)
    return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def _closest_gemini_aspect_ratio(width: int, height: int) -> str:
    supported = {
        "1:1": 1.0,
        "2:3": 2 / 3,
        "3:2": 3 / 2,
        "3:4": 3 / 4,
        "4:3": 4 / 3,
        "4:5": 4 / 5,
        "5:4": 5 / 4,
        "9:16": 9 / 16,
        "16:9": 16 / 9,
        "21:9": 21 / 9,
    }
    ratio = width / max(height, 1)
    return min(supported, key=lambda k: abs(supported[k] - ratio))


def unwrap_diptych_if_present(
    edited_bgr: np.ndarray,
    source_shape: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    """Gemini often returns before/after side-by-side; keep the 'after' panel."""
    sh, sw = source_shape[:2]
    h, w = edited_bgr.shape[:2]
    tol = 0.28 * max(sh, sw)

    if w >= int(sw * 1.55) and abs(h - sh) <= tol:
        panel = edited_bgr[:, w // 2 :]
        logger.info(
            "generative diptych: horizontal split detected (%sx%s), using right panel",
            w,
            h,
        )
        return panel, True
    if h >= int(sh * 1.55) and abs(w - sw) <= tol:
        panel = edited_bgr[h // 2 :, :]
        logger.info(
            "generative diptych: vertical split detected (%sx%s), using bottom panel",
            w,
            h,
        )
        return panel, True
    return edited_bgr, False


def is_gemini_model(model: str) -> bool:
    name = model.strip().lower()
    return "gemini" in name and "image" in name


def resolve_mask_policy(
    *,
    transport: str,
    model: str,
    use_mask: bool,
    composite_requested: bool,
) -> tuple[bool, bool]:
    """Return (send_mask_to_api, composite_locally).

    Gemini image models via OpenAI /images/edits do not support DALL-E-style RGBA masks.
    Local compositing also fails when the model returns a misaligned full re-generation.
    """
    if transport == "gemini_native":
        return False, False
    if transport == "custom_json":
        return use_mask, composite_requested
    if is_gemini_model(model):
        if use_mask:
            logger.info(
                "generative: skipping API mask for Gemini model on openai_images_edit "
                "(proxy does not support OpenAI RGBA inpaint reliably)"
            )
        if composite_requested:
            logger.info(
                "generative: skipping local composite for Gemini model "
                "(model may re-generate the whole face; compositing causes visible seams)"
            )
        return False, False
    return use_mask, composite_requested


def save_try_on_debug(
    run_dir,
    *,
    category: str,
    image_bgr: np.ndarray,
    mask: np.ndarray | None,
    zones: list[str],
    model_raw_bgr: np.ndarray | None = None,
    composite_applied: bool = False,
    api_use_mask: bool = False,
) -> None:
    from pathlib import Path

    if run_dir is None:
        return
    if not isinstance(run_dir, Path):
        return
    prefix = f"tryon_{category}"
    if mask is not None and int(np.max(mask)) > 0:
        cv2.imwrite(str(run_dir / f"{prefix}_mask.png"), mask)
        overlay = image_bgr.copy().astype(np.float32)
        tint = np.array([0.0, 0.0, 255.0], dtype=np.float32)
        sel = mask > 127
        overlay[sel] = overlay[sel] * 0.45 + tint * 0.55
        cv2.imwrite(str(run_dir / f"{prefix}_mask_overlay.jpg"), np.clip(overlay, 0, 255).astype(np.uint8))
        if api_use_mask:
            rgba = np.zeros((*mask.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = 255
            rgba[..., 3] = np.where(mask > 127, 0, 255).astype(np.uint8)
            cv2.imwrite(str(run_dir / f"{prefix}_openai_rgba_mask.png"), rgba)
    if model_raw_bgr is not None:
        cv2.imwrite(str(run_dir / f"{prefix}_model_raw.jpg"), model_raw_bgr)
    zone_text = ", ".join(zones) if zones else "none"
    (run_dir / f"{prefix}_mask_meta.txt").write_text(
        f"zones={zone_text}\n"
        f"api_use_mask={api_use_mask}\n"
        f"composite_applied={composite_applied}\n",
        encoding="utf-8",
    )


def composite_masked(
    original_bgr: np.ndarray,
    edited_bgr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Blend edited pixels only inside the feathered mask; preserve original elsewhere."""
    h, w = original_bgr.shape[:2]
    if edited_bgr.shape[:2] != (h, w):
        edited_bgr = cv2.resize(edited_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)
    if alpha.ndim == 2:
        alpha = alpha[..., np.newaxis]
    blended = original_bgr.astype(np.float32) * (1.0 - alpha) + edited_bgr.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _normalize_gemini_model(model: str) -> str:
    name = model.strip()
    if name.lower().startswith("google/"):
        return name.split("/", 1)[1]
    return name


def _build_multipart(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, bytes, str]],
) -> tuple[bytes, str]:
    boundary = f"----faceai{uuid.uuid4().hex}"
    lines: list[bytes] = []

    for name, value in fields:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        lines.append(value.encode("utf-8"))
        lines.append(b"\r\n")

    for name, filename, content, content_type in files:
        lines.append(f"--{boundary}\r\n".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        lines.append(f"Content-Type: {content_type}\r\n\r\n".encode())
        lines.append(content)
        lines.append(b"\r\n")

    lines.append(f"--{boundary}--\r\n".encode())
    body = b"".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {**headers, "Content-Type": "application/json"}
    req = request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"generative HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"generative request failed: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("generative HTTP body must be a JSON object")
    return parsed


def _http_post_multipart(
    url: str,
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, bytes, str]],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    body, content_type = _build_multipart(fields, files)
    hdrs = {**headers, "Content-Type": content_type}
    req = request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"generative HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"generative request failed: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("generative HTTP body must be a JSON object")
    return parsed


def resolve_generative_url(api_url: str, transport: str) -> str:
    url = api_url.strip().rstrip("/")
    if transport in ("custom_json", "gemini_native"):
        return url
    lower = url.lower()
    if lower.endswith("/images/edits"):
        return url
    if lower.endswith("/v1"):
        return f"{url}/images/edits"
    return f"{url}/v1/images/edits"


def resolve_openai_response_format(model: str, configured: str) -> str | None:
    cfg = configured.strip().lower()
    if cfg in ("", "omit", "none"):
        return None
    if cfg != "auto":
        return configured.strip()
    if "dall-e" in model.strip().lower():
        return "b64_json"
    return None


def _fetch_image_b64(url: str, timeout_s: float) -> str:
    req = request.Request(url, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
    except error.URLError as exc:
        raise RuntimeError(f"generative image download failed: {exc}") from exc
    return base64.b64encode(raw).decode("ascii")


def parse_openai_images_response(body: dict[str, Any], *, timeout_s: float = 90.0) -> str:
    data = body.get("data") or []
    if not data:
        raise ValueError("generative response missing data")
    first = data[0]
    if not isinstance(first, dict):
        raise ValueError("generative data[0] must be an object")
    b64 = first.get("b64_json")
    if isinstance(b64, str) and b64.strip():
        return b64
    url = first.get("url")
    if isinstance(url, str) and url.strip():
        return _fetch_image_b64(url.strip(), timeout_s)
    raise ValueError("generative response missing b64_json or url")


def parse_custom_json_response(body: dict[str, Any]) -> str:
    b64 = body.get("image_b64")
    if not isinstance(b64, str) or not b64.strip():
        raise ValueError("generative response missing image_b64")
    return b64


class GenerativeModelAPI:
    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        transport: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        strength: float | None = None,
        post_json: Callable[..., dict[str, Any]] | None = None,
        post_multipart: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.api_url = (api_url if api_url is not None else settings.generative_api_url).strip()
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = settings.resolved_generative_api_key
        self.transport = transport if transport is not None else settings.generative_transport
        self.model = model if model is not None else settings.generative_model
        self.timeout_s = timeout_s if timeout_s is not None else settings.generative_timeout_s
        self.strength = strength if strength is not None else settings.generative_strength
        self._post_json = post_json or _http_post_json
        self._post_multipart = post_multipart or _http_post_multipart

    @property
    def available(self) -> bool:
        url = self.api_url.strip().lower()
        if self.transport == "gemini_native":
            return bool(self.api_key)
        return bool(url) and url != "none"

    def render(
        self,
        image_bgr: np.ndarray,
        edit_mask: np.ndarray | None,
        prompt: str,
        *,
        negative_prompt: str = "",
        use_mask: bool = True,
        composite_mask: np.ndarray | None = None,
        debug_ctx: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        if not self.available:
            return None
        composite_requested = composite_mask is not None
        api_use_mask, do_composite = resolve_mask_policy(
            transport=self.transport,
            model=self.model,
            use_mask=use_mask,
            composite_requested=composite_requested,
        )

        if api_use_mask:
            if edit_mask is None or int(np.max(edit_mask)) < 1:
                logger.warning("generative render skipped: empty mask")
                return None
        elif edit_mask is not None and int(np.max(edit_mask)) < 1:
            edit_mask = None

        blend_mask = composite_mask if do_composite else None
        if blend_mask is None and do_composite and settings.generative_composite_mask:
            blend_mask = edit_mask
        if blend_mask is not None and int(np.max(blend_mask)) < 1:
            blend_mask = None

        if not self.api_key:
            logger.warning(
                "generative render skipped: no API key "
                "(set FACE_AI_GENERATIVE_API_KEY or FACE_AI_LLM_API_KEY)"
            )
            return None

        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(2):
            try:
                if self.transport == "custom_json":
                    b64 = self._render_custom_json(
                        image_bgr,
                        edit_mask,
                        prompt,
                        negative_prompt,
                        headers,
                        use_mask=api_use_mask,
                    )
                elif self.transport == "gemini_native":
                    b64 = self._render_gemini_native(
                        image_bgr,
                        prompt,
                        negative_prompt,
                    )
                else:
                    b64 = self._render_openai_edit(
                        image_bgr,
                        edit_mask,
                        prompt,
                        headers,
                        use_mask=api_use_mask,
                    )
                out = _decode_image_b64(b64)
                out, diptych = unwrap_diptych_if_present(out, image_bgr.shape)
                if debug_ctx is not None:
                    debug_ctx["diptych_unwrapped"] = diptych
                out = _align_to_source_size(out, image_bgr.shape)
                if debug_ctx is not None:
                    debug_ctx["model_raw_bgr"] = out.copy()
                    debug_ctx["api_use_mask"] = api_use_mask
                if blend_mask is not None:
                    out = composite_masked(image_bgr, out, blend_mask)
                    logger.info("generative composite: applied local mask blend")
                if debug_ctx is not None:
                    debug_ctx["composite_applied"] = blend_mask is not None
                return out
            except Exception as exc:
                logger.warning("generative render attempt %s failed: %s", attempt + 1, exc)
        return None

    def _render_openai_edit(
        self,
        image_bgr: np.ndarray,
        edit_mask: np.ndarray | None,
        prompt: str,
        headers: dict[str, str],
        *,
        use_mask: bool = True,
    ) -> str:
        url = resolve_generative_url(self.api_url, "openai_images_edit")
        image_png = _bgr_to_png_bytes(image_bgr)
        fields: list[tuple[str, str]] = [
            ("prompt", prompt),
            ("model", self.model),
            ("n", "1"),
            ("size", settings.generative_image_size),
        ]
        response_format = resolve_openai_response_format(
            self.model, settings.generative_response_format
        )
        if response_format:
            fields.append(("response_format", response_format))
        files = [("image", "image.png", image_png, "image/png")]
        if use_mask and edit_mask is not None:
            mask_png = _mask_to_openai_rgba(edit_mask, image_bgr.shape)
            files.append(("mask", "mask.png", mask_png, "image/png"))
        logger.info(
            "generative OpenAI edit: url=%s model=%s use_mask=%s response_format=%s",
            url,
            self.model,
            use_mask,
            response_format or "(omitted)",
        )
        try:
            body = self._post_multipart(url, fields, files, headers, self.timeout_s)
        except RuntimeError as exc:
            if response_format and "response_format" in str(exc):
                logger.info("generative: retrying without response_format")
                fields = [f for f in fields if f[0] != "response_format"]
                body = self._post_multipart(url, fields, files, headers, self.timeout_s)
            else:
                raise
        return parse_openai_images_response(body, timeout_s=self.timeout_s)

    def _render_custom_json(
        self,
        image_bgr: np.ndarray,
        edit_mask: np.ndarray | None,
        prompt: str,
        negative_prompt: str,
        headers: dict[str, str],
        *,
        use_mask: bool = True,
    ) -> str:
        url = resolve_generative_url(self.api_url, "custom_json")
        ok, img_buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise ValueError("failed to encode image as JPEG")
        image_b64 = base64.b64encode(img_buf.tobytes()).decode("ascii")
        payload: dict[str, Any] = {
            "image_b64": image_b64,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": self.strength,
        }
        if use_mask and edit_mask is not None:
            payload["mask_b64"] = base64.b64encode(
                _mask_to_png_bytes(edit_mask, image_bgr.shape)
            ).decode("ascii")
        logger.info("generative custom_json: url=%s use_mask=%s", url, use_mask)
        body = self._post_json(url, payload, headers, self.timeout_s)
        return parse_custom_json_response(body)

    def _render_gemini_native(
        self,
        image_bgr: np.ndarray,
        prompt: str,
        negative_prompt: str,
    ) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "gemini_native transport requires google-genai (pip install google-genai)"
            ) from exc

        from PIL import Image

        model = _normalize_gemini_model(self.model)
        api_url = self.api_url.strip().rstrip("/").lower()
        if api_url in ("none", ""):
            client = genai.Client(api_key=self.api_key)
        elif "generativelanguage.googleapis.com" in api_url:
            client = genai.Client(api_key=self.api_key)
        else:
            client = genai.Client(
                api_key=self.api_key,
                http_options={"base_url": self.api_url.strip().rstrip("/")},
            )

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        h, w = image_bgr.shape[:2]
        aspect = _closest_gemini_aspect_ratio(w, h)
        full_prompt = (
            "Edit the attached photograph in place. Return exactly one full-frame image.\n"
            f"{prompt.strip()}"
        )
        if negative_prompt.strip():
            full_prompt = f"{full_prompt}\n\nAvoid: {negative_prompt.strip()}"

        logger.info("generative gemini_native: model=%s aspect_ratio=%s", model, aspect)
        response = client.models.generate_content(
            model=model,
            contents=[pil_image, full_prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(aspect_ratio=aspect),
            ),
        )

        for part in response.parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                raw = inline.data
                if isinstance(raw, str):
                    raw = base64.b64decode(raw)
                arr = np.frombuffer(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    if not ok:
                        raise ValueError("failed to encode gemini result")
                    return base64.b64encode(buf.tobytes()).decode("ascii")
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                try:
                    out_pil = part.as_image()
                    if out_pil is not None:
                        out_bgr = cv2.cvtColor(np.array(out_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
                        ok, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                        if not ok:
                            raise ValueError("failed to encode gemini result")
                        return base64.b64encode(buf.tobytes()).decode("ascii")
                except Exception:
                    pass

        raise ValueError("gemini response missing image data")
