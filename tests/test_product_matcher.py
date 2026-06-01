import pytest

from app.services.product_catalog import get_product, list_products, reload_catalog
from app.services.product_matcher import delta_e_to_match_pct, match_products, season_zone_lab


@pytest.fixture(autouse=True)
def _fresh_catalog():
    reload_catalog()
    yield
    reload_catalog()


def test_catalog_loads_30_products():
    products = list_products()
    assert len(products) == 30
    categories = {p.category for p in products}
    assert categories == {"lipstick", "blush", "foundation", "frames", "hairstyle"}


def test_get_product_by_sku():
    p = get_product("mac-ruby-woo")
    assert p is not None
    assert p.brand == "MAC"


def test_match_lipstick_light_summer():
    result = match_products("light_summer", "lipstick", top_k=3)
    assert result.season_twelve == "light_summer"
    assert len(result.matches) == 3
    assert result.matches[0].sku == "charlotte-pillow-talk"
    assert result.matches[0].delta_e <= result.matches[1].delta_e
    assert all(m.match_pct >= 0 for m in result.matches)


def test_match_expanded_when_no_exact_tag():
    result = match_products("soft_summer", "lipstick", top_k=3)
    assert len(result.matches) == 3


def test_match_with_target_lab():
    target = (65.0, 18.0, 8.0)
    result = match_products("light_summer", "lipstick", target_lab=target, top_k=1)
    assert result.target_lab == target
    assert len(result.matches) == 1


def test_delta_e_to_match_pct_bounds():
    assert delta_e_to_match_pct(0.0) == 100.0
    assert delta_e_to_match_pct(100.0) == 0.0


def test_makeup_db_season_zone():
    lab = season_zone_lab("true_winter", "lips")
    assert lab is not None
    assert lab[0] == pytest.approx(38.0)
    assert season_zone_lab("unknown_season", "lips") is None
