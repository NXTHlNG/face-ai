"""Ensemble fusion with Park IMCOM'18 contributor."""

from app.pipeline.ensemble import fuse_seasonal


def _munsell_stub(four: str, sixteen: str = "true_light", conf: float = 0.7) -> dict:
    return {
        "seasonal_sixteen": sixteen,
        "seasonal_guess": four,
        "confidence": conf,
        "munsell_scores": {"undertone": 3, "chroma": 3, "value": 3, "contrast": 3},
    }


def _park_stub(four: str, twelve: str, conf: float = 0.65) -> dict:
    return {
        "seasonal_guess": four,
        "seasonal_twelve": twelve,
        "seasonal_confidence": conf,
        "classifier_contributors": {
            "park_imcom18": {"four": four, "confidence": conf},
        },
    }


def test_ensemble_includes_park_vote():
    out = fuse_seasonal(
        _munsell_stub("summer"),
        {"seasonal_guess": "summer", "confidence": 0.2},
        park=_park_stub("winter", "true_winter"),
    )
    assert out["seasonal_method"] == "ensemble"
    assert "park_imcom18" in out["classifier_contributors"]
    assert out["seasonal_guess"] in {"summer", "winter"}


def test_ensemble_park_unknown_skipped():
    out = fuse_seasonal(
        _munsell_stub("autumn"),
        {"seasonal_guess": "autumn", "confidence": 0.2},
        park={"seasonal_guess": "unknown", "seasonal_confidence": 0.2},
    )
    assert "park_imcom18" not in out["classifier_contributors"]
