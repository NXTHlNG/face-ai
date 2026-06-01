from __future__ import annotations

import json
import math

from app.backends.color.delta_e import rgb_to_lab
from app.config import settings

_ANCHORS: dict | None = None


def _load_anchors() -> dict:
    global _ANCHORS
    if _ANCHORS is None:
        raw = json.loads(
            (settings.data_dir / "reference_swatches.json").read_text(encoding="utf-8")
        )
        _ANCHORS = raw["season_rgb_anchors"]
    return _ANCHORS


def vote_four_seasons(skin_rgb: tuple[float, float, float] | None) -> dict:
    if skin_rgb is None:
        return {
            "seasonal_guess": "unknown",
            "confidence": 0.15,
            "votes": {},
        }
    anchors = _load_anchors()
    skin_lab = rgb_to_lab(*skin_rgb)
    votes: dict[str, float] = {}
    for season, triplets in anchors.items():
        dists = []
        for rgb in triplets:
            ref_lab = rgb_to_lab(rgb[0], rgb[1], rgb[2])
            d = math.sqrt(
                (skin_lab[0] - ref_lab[0]) ** 2
                + (skin_lab[1] - ref_lab[1]) ** 2
                + (skin_lab[2] - ref_lab[2]) ** 2
            )
            dists.append(float(d))
        votes[season] = -min(dists)
    ranked = sorted(votes.items(), key=lambda x: -x[1])
    best = ranked[0][0]
    margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
    conf = float(min(0.45, max(0.12, 0.2 + margin * 0.05)))
    total = sum(max(v, 0.0) for v in votes.values()) + 1e-9
    return {
        "seasonal_guess": best,
        "confidence": conf,
        "votes": votes,
        "top_k_four": [
            {"season": k, "probability": round(max(v, 0) / total, 4)}
            for k, v in ranked[:2]
        ],
    }
