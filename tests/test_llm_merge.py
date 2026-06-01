import pytest

from app.backends.llm.schema_merge import (
    apply_llm_seasonal,
    extract_json_object,
    parse_llm_payload,
)
from app.schemas.analysis import RecommendationItem, SeasonalAnalysis


def _cv_seasonal() -> SeasonalAnalysis:
    return SeasonalAnalysis(
        seasonal_guess="summer",
        seasonal_twelve="true_summer",
        seasonal_sixteen="cool_summer",
        seasonal_confidence=0.55,
        seasonal_twelve_confidence=0.52,
        undertone_hint="cool",
        seasonal_method="ensemble",
    )


def _rule_recs() -> list[RecommendationItem]:
    return [
        RecommendationItem(
            category="general",
            title="Rules fallback",
            detail="From rules engine.",
            rule_id="general_disclaimer_multi_photo",
            rule_version="2.0.0",
        )
    ]


def test_parse_llm_payload_valid():
    payload = parse_llm_payload(
        {
            "seasonal_guess": "winter",
            "seasonal_twelve": "true_winter",
            "seasonal_sixteen": "cool_winter",
            "seasonal_confidence": 0.81,
            "seasonal_twelve_confidence": 0.79,
            "undertone_hint": "cool",
            "recommendations": [{"category": "makeup", "title": "ignored", "detail": "ignored"}],
        }
    )
    assert payload.seasonal_twelve == "true_winter"


def test_extract_json_from_markdown_fence():
    raw = 'Here is JSON:\n```json\n{"seasonal_guess":"spring","seasonal_twelve":"light_spring","seasonal_sixteen":"light_spring","seasonal_confidence":0.7,"seasonal_twelve_confidence":0.68,"undertone_hint":"warm"}\n```'
    data = extract_json_object(raw)
    assert data["seasonal_twelve"] == "light_spring"


def test_apply_llm_overrides_season_only():
    llm = {
        "seasonal_guess": "winter",
        "seasonal_twelve": "bright_winter",
        "seasonal_sixteen": "bright_winter",
        "seasonal_confidence": 0.88,
        "seasonal_twelve_confidence": 0.86,
        "undertone_hint": "cool",
        "recommendations": [
            {
                "category": "clothing_colors",
                "title": "Палитра",
                "detail": "Should be ignored.",
            }
        ],
    }
    seasonal, used = apply_llm_seasonal(_cv_seasonal(), llm)

    assert used is True
    assert seasonal.seasonal_twelve == "bright_winter"
    assert seasonal.seasonal_method == "llm"


def test_apply_llm_invalid_json_falls_back_to_cv():
    seasonal, used = apply_llm_seasonal(_cv_seasonal(), "not json")

    assert used is False
    assert seasonal.seasonal_twelve == "true_summer"


def test_apply_llm_does_not_touch_recommendations():
    llm = {
        "seasonal_guess": "autumn",
        "seasonal_twelve": "soft_autumn",
        "seasonal_sixteen": "soft_autumn",
        "seasonal_confidence": 0.75,
        "seasonal_twelve_confidence": 0.73,
        "undertone_hint": "warm",
    }
    seasonal, used = apply_llm_seasonal(_cv_seasonal(), llm)

    assert used is True
    assert seasonal.seasonal_twelve == "soft_autumn"
    assert _rule_recs()[0].rule_id == "general_disclaimer_multi_photo"
