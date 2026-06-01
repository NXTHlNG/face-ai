from __future__ import annotations

from app.backends.season.season_maps import SIXTEEN_TO_TWELVE, TWELVE_TO_FOUR

NEIGHBOR_PAIRS = {
    frozenset({"spring", "summer"}),
    frozenset({"spring", "autumn"}),
    frozenset({"summer", "winter"}),
    frozenset({"autumn", "winter"}),
}


def _map_sixteen_to_twelve(sixteen: str) -> str:
    if sixteen in SIXTEEN_TO_TWELVE:
        return SIXTEEN_TO_TWELVE[sixteen]
    if sixteen in TWELVE_TO_FOUR:
        return sixteen
    return "unknown"


def fuse_seasonal(
    munsell: dict,
    swatch: dict,
    *,
    park: dict | None = None,
    wrist_prior: dict | None = None,
) -> dict:
    weights = {
        "munsell": 0.70,
        "park": 0.15,
        "swatch": 0.10,
        "wrist": 0.05,
    }

    sixteen_scores: dict[str, float] = {}
    twelve_scores: dict[str, float] = {}
    four_scores: dict[str, float] = {}

    def add_16(tid: str, w: float, conf: float) -> None:
        if tid == "unknown":
            return
        sixteen_scores[tid] = sixteen_scores.get(tid, 0.0) + w * conf

    def add_12(tid: str, w: float, conf: float) -> None:
        if tid == "unknown":
            return
        twelve_scores[tid] = twelve_scores.get(tid, 0.0) + w * conf
        parent = TWELVE_TO_FOUR.get(tid)
        if parent:
            four_scores[parent] = four_scores.get(parent, 0.0) + w * conf

    m16 = munsell.get("seasonal_sixteen", "unknown")
    m4 = munsell.get("seasonal_guess", "unknown")
    mc = float(munsell.get("confidence", 0.3))
    add_16(m16, weights["munsell"], mc)
    m12 = _map_sixteen_to_twelve(m16)
    add_12(m12, weights["munsell"] * 0.9, mc)
    four_scores[m4] = four_scores.get(m4, 0.0) + weights["munsell"] * mc

    if park and park.get("seasonal_guess", "unknown") != "unknown":
        p4 = park["seasonal_guess"]
        p12 = park.get("seasonal_twelve", "unknown")
        pc = float(park.get("seasonal_confidence", 0.3))
        add_12(p12, weights["park"], pc)
        four_scores[p4] = four_scores.get(p4, 0.0) + weights["park"] * pc

    s4 = swatch.get("seasonal_guess", "unknown")
    sc = float(swatch.get("confidence", 0.15))
    four_scores[s4] = four_scores.get(s4, 0.0) + weights["swatch"] * sc

    if wrist_prior:
        wu = wrist_prior.get("undertone_hint")
        wc = float(wrist_prior.get("confidence", 0.5))
        if wu == "warm":
            for s in ("spring", "autumn"):
                four_scores[s] = four_scores.get(s, 0.0) + weights["wrist"] * wc
        elif wu == "cool":
            for s in ("summer", "winter"):
                four_scores[s] = four_scores.get(s, 0.0) + weights["wrist"] * wc

    if not sixteen_scores:
        best_16 = "unknown"
    else:
        best_16 = max(sixteen_scores, key=sixteen_scores.get)
    if not twelve_scores:
        best_12 = "unknown"
    else:
        best_12 = max(twelve_scores, key=twelve_scores.get)
    if not four_scores:
        best_4 = "unknown"
    else:
        best_4 = max(four_scores, key=four_scores.get)

    total_4 = sum(four_scores.values()) + 1e-9
    ranked_4 = sorted(four_scores.items(), key=lambda x: -x[1])
    ranked_12 = sorted(twelve_scores.items(), key=lambda x: -x[1])

    margin_4 = 0.0
    if len(ranked_4) >= 2:
        margin_4 = ranked_4[0][1] - ranked_4[1][1]
    conf = 0.32 + min(0.5, margin_4 / total_4 * 2.5)
    if len(ranked_4) >= 2:
        pair = frozenset({ranked_4[0][0], ranked_4[1][0]})
        if pair in NEIGHBOR_PAIRS:
            conf = min(0.9, conf + 0.06)

    notes: list[str] = []
    if len(ranked_4) >= 2:
        pair = frozenset({ranked_4[0][0], ranked_4[1][0]})
        if pair in NEIGHBOR_PAIRS:
            notes.append("borderline_neighbor_season")

    contributors: dict = {
        "munsell": {
            "sixteen": m16,
            "four": m4,
            "confidence": mc,
        },
        "swatch_vote": {
            "four": s4,
            "confidence": sc,
        },
    }
    if park and park.get("seasonal_guess", "unknown") != "unknown":
        contributors["park_imcom18"] = park.get("classifier_contributors", {}).get(
            "park_imcom18",
            {
                "four": park["seasonal_guess"],
                "twelve": park.get("seasonal_twelve"),
                "confidence": float(park.get("seasonal_confidence", 0.3)),
            },
        )

    return {
        "seasonal_sixteen": best_16,
        "seasonal_twelve": best_12,
        "seasonal_guess": best_4,
        "seasonal_confidence": round(float(min(0.9, max(0.2, conf))), 4),
        "seasonal_twelve_confidence": round(float(min(0.9, max(0.2, conf * 0.95))), 4),
        "seasonal_method": "ensemble",
        "munsell_scores": munsell.get("munsell_scores"),
        "classifier_contributors": contributors,
        "seasonal_guess_top_k": [
            {"season": k, "probability": round(v / total_4, 4)}
            for k, v in ranked_4[:2]
        ],
        "seasonal_twelve_top_k": [
            {"subtype": k, "probability": round(v / (sum(twelve_scores.values()) + 1e-9), 4)}
            for k, v in ranked_12[:3]
        ],
        "seasonal_notes": notes,
    }
