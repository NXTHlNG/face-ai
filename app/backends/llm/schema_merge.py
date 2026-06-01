from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.analysis import SeasonalAnalysis

_SEASONAL_GUESS = Literal["spring", "summer", "autumn", "winter", "unknown"]
_SEASONAL_TWELVE = Literal[
    "light_spring",
    "true_spring",
    "bright_spring",
    "light_summer",
    "true_summer",
    "soft_summer",
    "soft_autumn",
    "true_autumn",
    "deep_autumn",
    "deep_winter",
    "true_winter",
    "bright_winter",
    "unknown",
]


class LLMAnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    seasonal_guess: _SEASONAL_GUESS
    seasonal_twelve: _SEASONAL_TWELVE
    seasonal_sixteen: str = "unknown"
    seasonal_confidence: float = Field(..., ge=0, le=1)
    seasonal_twelve_confidence: float = Field(..., ge=0, le=1)
    undertone_hint: Literal["warm", "cool", "neutral"] = "neutral"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty LLM response")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped, re.IGNORECASE)
    if fence:
        stripped = fence.group(1).strip()

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON root must be an object")
    return parsed


def parse_llm_payload(raw: str | dict[str, Any]) -> LLMAnalysisPayload:
    data = raw if isinstance(raw, dict) else extract_json_object(raw)
    return LLMAnalysisPayload.model_validate(data)


def merge_seasonal_from_llm(
    cv_seasonal: SeasonalAnalysis,
    llm: LLMAnalysisPayload,
) -> SeasonalAnalysis:
    return cv_seasonal.model_copy(
        update={
            "seasonal_guess": llm.seasonal_guess,
            "seasonal_twelve": llm.seasonal_twelve,
            "seasonal_sixteen": llm.seasonal_sixteen or cv_seasonal.seasonal_sixteen,
            "seasonal_confidence": llm.seasonal_confidence,
            "seasonal_twelve_confidence": llm.seasonal_twelve_confidence,
            "undertone_hint": llm.undertone_hint,
            "seasonal_method": "llm",
        }
    )


def apply_llm_seasonal(
    cv_seasonal: SeasonalAnalysis,
    llm_raw: str | dict[str, Any] | None,
) -> tuple[SeasonalAnalysis, bool]:
    if llm_raw is None:
        return cv_seasonal, False

    try:
        payload = parse_llm_payload(llm_raw)
    except (ValidationError, ValueError, json.JSONDecodeError):
        return cv_seasonal, False

    return merge_seasonal_from_llm(cv_seasonal, payload), True
