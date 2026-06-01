from app.backends.season.aua_scales import (
    lab_chroma_to_aua,
    lab_luminance_contrast_aua,
    normalize_chroma_aua,
    normalize_contrast_aua,
    normalize_value_aua,
    opencv_L_to_aua_value,
)


def test_lab_chroma_to_aua_mapping():
    assert lab_chroma_to_aua(0.0) == 0.0
    assert lab_chroma_to_aua(8.0) == 0.2
    assert lab_chroma_to_aua(32.0) == 0.8
    assert lab_chroma_to_aua(80.0) == 1.0


def test_normalize_chroma_aua_thresholds():
    assert normalize_chroma_aua(0.19) == 1
    assert normalize_chroma_aua(0.2) == 2
    assert normalize_chroma_aua(0.39) == 2
    assert normalize_chroma_aua(0.79) == 4
    assert normalize_chroma_aua(0.95) == 5


def test_normalize_value_aua_from_opencv_L():
    assert normalize_value_aua(opencv_L_to_aua_value(51)) == 2
    assert normalize_value_aua(opencv_L_to_aua_value(127)) == 3
    assert normalize_value_aua(opencv_L_to_aua_value(220)) == 5


def test_contrast_aua_high_ratio_for_light_skin_dark_hair():
    # OpenCV L: light skin ~200, dark hair ~35 → ratio ~5.7 → bin 2
    ratio = lab_luminance_contrast_aua(35.0, 200.0)
    assert 5.0 < ratio < 9.0
    assert normalize_contrast_aua(ratio) == 2


def test_contrast_aua_extreme_winter_like():
    ratio = lab_luminance_contrast_aua(12.0, 250.0)
    assert ratio >= 17.0
    assert normalize_contrast_aua(ratio) == 5


def test_contrast_aua_low_similar_zones():
    ratio = lab_luminance_contrast_aua(68.0, 72.0)
    assert ratio < 1.1
    assert normalize_contrast_aua(ratio) == 1
