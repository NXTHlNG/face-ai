"""Face undertone inference (skin + hair anchors, iris makeup guard)."""

from __future__ import annotations

# Borderline |a*-b*| on skin: lean cool when contrast is low (summer over spring).
SKIN_AB_BORDERLINE = 2.5
# Fair porcelain: rosy a* without strong golden b*.
FAIR_COOL_A_MIN = 8.0
FAIR_COOL_B_MAX = 14.0


def skin_undertone_from_ab(skin_ab: tuple[float, float]) -> str:
    """Undertone from skin a*/b* only (Park rule + fair pink-cool heuristic)."""
    a_star, b_star = skin_ab
    if a_star > b_star:
        return "cool"
    if a_star >= FAIR_COOL_A_MIN and b_star < FAIR_COOL_B_MAX:
        return "cool"
    margin = abs(a_star - b_star)
    if margin < SKIN_AB_BORDERLINE and a_star > 6.0:
        return "cool"
    return "warm"


def park_undertone_from_ab(skin_ab: tuple[float, float]) -> str:
    """Park IMCOM'18 API — delegates to ``skin_undertone_from_ab``."""
    return skin_undertone_from_ab(skin_ab)


def iris_undertone_untrusted(
    hair_ab: tuple[float, float] | None,
    iris_ab: tuple[float, float] | None,
) -> bool:
    """True when iris colour likely comes from eyeshadow / lid, not the iris."""
    if hair_ab is None or iris_ab is None:
        return False
    ha, hb = hair_ab
    ia, ib = iris_ab
    hair_ash = ha < 6.0 and hb < 10.0
    iris_warm_beige = ib > 9.0 and ia > 4.0
    return hair_ash and iris_warm_beige


def infer_face_undertone(
    skin_ab: tuple[float, float],
    hair_ab: tuple[float, float] | None = None,
    iris_ab: tuple[float, float] | None = None,
    *,
    low_contrast: bool = False,
) -> str:
    """Fuse skin, hair, and optional iris into warm / cool / neutral."""
    skin_ut = skin_undertone_from_ab(skin_ab)
    warm_s = 2.4 if skin_ut == "warm" else 0.0
    cool_s = 2.4 if skin_ut == "cool" else 0.0
    if skin_ut not in ("warm", "cool"):
        warm_s += 0.8
        cool_s += 0.8

    a_star, b_star = skin_ab
    if low_contrast and abs(a_star - b_star) < SKIN_AB_BORDERLINE:
        cool_s += 0.9

    if hair_ab is not None:
        ha, hb = hair_ab
        if ha < 6.0 and hb < 10.0:
            cool_s += 2.4
        elif hb > 14.0 and ha > -3.0:
            warm_s += 1.9
        elif hb < -2.0 or ha < -5.0:
            cool_s += 1.3

    if iris_ab is not None and not iris_undertone_untrusted(hair_ab, iris_ab):
        ia, ib = iris_ab
        if ib > 3.5 and ia > -2.0:
            warm_s += 1.0 * min(max(ib, 0.0) / 13.5, 1.0)
        elif ib < -2.5 or ia < -4.0:
            cool_s += 1.0 * min(max(-ib, 0.0) / 13.5, 1.0)

    if warm_s > cool_s * 1.12:
        return "warm"
    if cool_s > warm_s * 1.12:
        return "cool"
    return "neutral"
