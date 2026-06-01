from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from app.backends.parsing import (
    bisenet_resnet34,
    farl_b,
    landmark_fallback,
    segface,
)
from app.backends.parsing.types import ParsingResult
from app.config import settings

_log = logging.getLogger(__name__)

ParsingBackendId = Literal[
    "bisenet_resnet34",
    "farl_b",
    "segface",
    "landmark_fallback",
]

_BACKENDS = {
    "bisenet_resnet34": bisenet_resnet34,
    "farl_b": farl_b,
    "segface": segface,
}


def _backend_error_hint(bid: str) -> str | None:
    mod = _BACKENDS.get(bid)
    if mod is not None and hasattr(mod, "last_error"):
        return mod.last_error()
    return None


def parsing_health() -> dict:
    health = {
        "bisenet_resnet34": bisenet_resnet34.is_available(),
        "farl_b": farl_b.is_available(),
        "segface": segface.is_available()
        and (segface._resolve_weights() is not None),  # noqa: SLF001
        "landmark_fallback": True,
    }
    hints: dict[str, str] = {}
    for bid in ("farl_b", "segface"):
        if not health.get(bid):
            err = _backend_error_hint(bid)
            if err:
                hints[bid] = err
            elif bid == "farl_b":
                hints["farl_b"] = (
                    "install requirements-parsing-ml.txt (torch, pyfacer)"
                )
            elif bid == "segface":
                hints["segface"] = (
                    "run scripts/download_segface_weights.py "
                    f"(models/segface/<model>_celeba_<size>/model_299.pt)"
                )
    return {"available": health, "hints": hints}


def _build_chain(requested: ParsingBackendId) -> list[ParsingBackendId]:
    chain: list[ParsingBackendId] = [requested]
    if requested != "landmark_fallback":
        for part in settings.parsing_fallback_chain.split(","):
            bid = part.strip()
            if bid in _BACKENDS and bid not in chain:
                chain.append(bid)  # type: ignore[arg-type]
    if "landmark_fallback" not in chain:
        chain.append("landmark_fallback")
    return chain


def parse_face(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    *,
    backend: ParsingBackendId | None = None,
) -> tuple[ParsingResult, ParsingBackendId, list[str]]:
    """
    Returns (result, requested_backend, notes).
    notes explains fallback when requested backend did not run.
    """
    requested = backend or settings.parsing_backend
    chain = _build_chain(requested)
    notes: list[str] = []

    for bid in chain:
        if bid == "landmark_fallback":
            if bid != requested:
                notes.append(f"parsing_fallback_to_{bid}")
            return landmark_fallback.parse(image_rgb, landmarks_px), requested, notes

        mod = _BACKENDS.get(bid)
        if mod is None:
            continue

        if hasattr(mod, "is_available") and not mod.is_available():
            if bid == requested:
                err = _backend_error_hint(bid)
                notes.append(
                    f"parsing_requested_{bid}_unavailable"
                    + (f":{err}" if err else "")
                )
            continue

        result = mod.parse(image_rgb, landmarks_px)
        if result is not None:
            if bid != requested:
                notes.append(f"parsing_fallback_{requested}_to_{bid}")
            return result, requested, notes

        if bid == requested:
            err = _backend_error_hint(bid)
            notes.append(
                f"parsing_{bid}_failed"
                + (f":{err}" if err else "")
            )

    notes.append(f"parsing_fallback_{requested}_to_landmark_fallback")
    return landmark_fallback.parse(image_rgb, landmarks_px), requested, notes


# Backward-compatible wrapper
def parse_face_legacy(
    image_rgb: np.ndarray,
    landmarks_px: np.ndarray,
    *,
    backend: ParsingBackendId | None = None,
) -> ParsingResult:
    result, _, _ = parse_face(image_rgb, landmarks_px, backend=backend)
    return result
