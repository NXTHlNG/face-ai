"""Unit tests for Park et al. IMCOM'18 seasonal formulas."""

from app.backends.season.park_imcom18 import (
    classify_park_imcom18,
    park_four_season,
    park_mean_contrast_L,
    park_undertone,
)


def test_park_undertone_cool_when_a_greater_than_b():
    assert park_undertone((5.0, -2.0)) == "cool"
    assert park_undertone((0.0, -5.0)) == "cool"


def test_park_undertone_warm_when_b_greater_or_equal_a():
    assert park_undertone((-2.0, 5.0)) == "warm"
    assert park_undertone((3.0, 3.0)) == "warm"


def test_park_undertone_fair_pink_borderline_is_cool():
    assert park_undertone((11.0, 12.0)) == "cool"


def test_park_contrast_mean_pairwise_cielab():
    # OpenCV L: skin=178 (~69.8), hair=102 (~40), iris=140 (~54.9)
    avg, regions = park_mean_contrast_L(178.0, 102.0, 140.0, use_cielab_scale=True)
    assert regions == ["skin", "hair", "iris"]
    assert 13.0 <= avg <= 35.0


def test_park_four_season_matrix():
    assert park_four_season("warm", 15.0)[0] == "spring"
    assert park_four_season("warm", 10.0)[0] == "autumn"
    assert park_four_season("cool", 15.0)[0] == "winter"
    assert park_four_season("cool", 10.0)[0] == "summer"


def test_classify_park_imcom18_full():
    out = classify_park_imcom18(
        skin_L=178.0,
        skin_ab=(2.0, 8.0),
        hair_L=102.0,
        iris_L=140.0,
        contrast_threshold=13.0,
    )
    assert out["seasonal_method"] == "park_imcom18"
    assert out["undertone_hint"] == "warm"
    assert out["seasonal_guess"] in {"spring", "autumn", "summer", "winter"}
    assert out["seasonal_twelve"] != "unknown"
    assert out["park_metrics"]["regions_used"] == ["skin", "hair", "iris"]


def test_classify_park_insufficient_regions():
    out = classify_park_imcom18(skin_L=170.0, skin_ab=(1.0, 4.0), hair_L=None, iris_L=None)
    assert out["seasonal_guess"] == "unknown"
    assert "park_insufficient_regions" in out["seasonal_notes"]
