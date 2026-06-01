from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any, Callable
from urllib import error, request

import cv2
import numpy as np

from app.backends.llm.schema_merge import LLMAnalysisPayload, parse_llm_payload
from app.config import settings

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "analysis_system.txt"
_DEFAULT_MODEL = "google/gemini-3.1-flash-lite-preview"
_DEFAULT_TIMEOUT_S = 60.0


def load_analysis_system_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _encode_image_jpeg_b64(image_bgr: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_message_content(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("LLM response missing choices")
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        content = "".join(parts)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response missing text content")
    return content


class LLMAnalysisAdapter:
    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        post_json: Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]] | None = None,
    ) -> None:
        self.api_url = (api_url if api_url is not None else settings.llm_api_url).strip()
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model
        self.timeout_s = timeout_s
        self._post_json = post_json or _http_post_json

    @property
    def available(self) -> bool:
        return settings.llm_enabled and bool(self.api_url)

    def analyze(
        self,
        image_bgr: np.ndarray,
        system_prompt: str | None = None,
    ) -> LLMAnalysisPayload | None:
        if not self.available:
            return None

        prompt = system_prompt or load_analysis_system_prompt()
        image_b64 = _encode_image_jpeg_b64(image_bgr)

        for attempt in range(2):
            try:
                raw_text = self._call_chat(prompt, image_b64)
                logger.info("LLM raw response (attempt %s): %s", attempt + 1, raw_text)
                payload = parse_llm_payload(raw_text)
                logger.info(
                    "LLM parsed seasonal: guess=%s twelve=%s sixteen=%s confidence=%.2f undertone=%s",
                    payload.seasonal_guess,
                    payload.seasonal_twelve,
                    payload.seasonal_sixteen,
                    payload.seasonal_confidence,
                    payload.undertone_hint,
                )
                return payload
            except Exception as exc:
                logger.warning("LLM analyze attempt %s failed: %s", attempt + 1, exc)
        logger.warning("LLM analyze exhausted retries; CV/rules season will be used")
        return None

    def _call_chat(self, system_prompt: str, image_b64: str) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this face photo and return the JSON object.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                },
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        logger.info("LLM request: model=%s url=%s", self.model, self.api_url)
        body = self._post_json(self.api_url, payload, headers, self.timeout_s)
        return _extract_message_content(body)


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM HTTP body must be a JSON object")
    return parsed
