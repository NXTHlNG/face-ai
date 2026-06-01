#!/usr/bin/env python3
"""Benchmark face-ai seasonal color analysis on Deep Armocromia RGB dataset.

Evaluates the full analysis pipeline (landmarks → parsing → color → season),
not an isolated DL classifier.

Usage:
  python scripts/eval_season_dataset.py
  python scripts/eval_season_dataset.py --split test --limit 50
  python scripts/eval_season_dataset.py --output artefacts/eval/deep_armocromia_test.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Benchmark should analyze all faces, not skip on photo-quality gate.
os.environ.setdefault("FACE_AI_SKIP_ANALYSIS_IF_PHOTO_POOR", "false")
os.environ.setdefault("FACE_AI_MASK_PREVIEW_ENABLED", "false")
os.environ.setdefault("FACE_AI_DEBUG_SAVE_IMAGES", "false")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from app.config import settings
from app.eval.deep_armocromia import (
    FOUR_SEASONS,
    TWELVE_SEASONS,
    DatasetSample,
    iter_samples,
    resolve_dataset_root,
)
from app.pipeline.orchestrator import AnalyzeOptions, analyze_bgr
from app.schemas.analysis import BackendOverrides


def _top_k_labels(items: list, key: str) -> list[str]:
    out: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            val = item.get(key)
        else:
            val = getattr(item, key, None)
        if val:
            out.append(str(val))
    return out


def _in_top_k(pred: str, truth: str, top_k: list[str], k: int) -> bool:
    if pred == truth:
        return True
    return truth in top_k[:k]


def _confusion(labels: list[str], truths: list[str], preds: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {t: {p: 0 for p in labels} for t in labels}
    for truth, pred in zip(truths, preds, strict=True):
        if truth in matrix and pred in matrix[truth]:
            matrix[truth][pred] += 1
    return matrix


def _per_class_accuracy(truths: list[str], preds: list[str]) -> dict[str, dict[str, float | int]]:
    by_class: dict[str, list[bool]] = defaultdict(list)
    for truth, pred in zip(truths, preds, strict=True):
        by_class[truth].append(pred == truth)
    return {
        cls: {
            "support": len(correct),
            "accuracy": round(sum(correct) / len(correct), 4) if correct else 0.0,
        }
        for cls, correct in sorted(by_class.items())
    }


@dataclass
class SampleOutcome:
    sample: DatasetSample
    status: str
    pred_four: str | None = None
    pred_twelve: str | None = None
    top_four: list[str] | None = None
    top_twelve: list[str] | None = None
    parsing_used: bool | None = None
    confidence_four: float | None = None
    confidence_twelve: float | None = None
    notes: list[str] | None = None


def analyze_sample(sample: DatasetSample, *, parsing_backend: str | None) -> SampleOutcome:
    bgr = cv2.imread(str(sample.path))
    if bgr is None:
        return SampleOutcome(sample=sample, status="read_error")

    opts = AnalyzeOptions()
    if parsing_backend:
        opts.backends = BackendOverrides(parsing_backend=parsing_backend)

    try:
        resp = analyze_bgr(bgr, opts)
    except Exception as exc:  # noqa: BLE001 — benchmark should continue
        return SampleOutcome(sample=sample, status="error", notes=[str(exc)])

    if resp.metrics is None:
        reason = "no_face" if "no_face" in (resp.confidence.notes or []) else "no_metrics"
        return SampleOutcome(
            sample=sample,
            status=reason,
            notes=list(resp.confidence.notes or []),
        )

    seasonal = resp.metrics.seasonal
    top_four = _top_k_labels(seasonal.seasonal_guess_top_k, "season")
    top_twelve = _top_k_labels(seasonal.seasonal_twelve_top_k, "subtype")

    pred_four = seasonal.seasonal_guess
    pred_twelve = seasonal.seasonal_twelve
    if pred_four not in FOUR_SEASONS:
        pred_four = "unknown"
    if pred_twelve not in TWELVE_SEASONS:
        pred_twelve = "unknown"

    if pred_four not in top_four:
        top_four = [pred_four, *top_four]
    if pred_twelve not in top_twelve:
        top_twelve = [pred_twelve, *top_twelve]

    return SampleOutcome(
        sample=sample,
        status="ok",
        pred_four=pred_four,
        pred_twelve=pred_twelve,
        top_four=top_four,
        top_twelve=top_twelve,
        parsing_used=resp.metrics.contrast.parsing_used,
        confidence_four=seasonal.seasonal_confidence,
        confidence_twelve=seasonal.seasonal_twelve_confidence,
        notes=list(resp.confidence.notes or []),
    )


def _accuracy(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def aggregate(outcomes: list[SampleOutcome]) -> dict:
    ok = [o for o in outcomes if o.status == "ok"]
    status_counts = dict(Counter(o.status for o in outcomes))
    analyzed = status_counts.get("ok", 0)

    truths_four = [o.sample.season_four for o in ok]
    preds_four = [o.pred_four or "unknown" for o in ok]
    truths_twelve = [o.sample.season_twelve for o in ok]
    preds_twelve = [o.pred_twelve or "unknown" for o in ok]

    acc4_1 = [p == t for t, p in zip(truths_four, preds_four, strict=True)]
    acc4_2 = [
        _in_top_k(p, t, o.top_four or [], 2)
        for t, p, o in zip(truths_four, preds_four, ok, strict=True)
    ]
    acc12_1 = [p == t for t, p in zip(truths_twelve, preds_twelve, strict=True)]
    acc12_2 = [
        _in_top_k(p, t, o.top_twelve or [], 2)
        for t, p, o in zip(truths_twelve, preds_twelve, ok, strict=True)
    ]
    acc12_3 = [
        _in_top_k(p, t, o.top_twelve or [], 3)
        for t, p, o in zip(truths_twelve, preds_twelve, ok, strict=True)
    ]

    parsing_rate = _accuracy([bool(o.parsing_used) for o in ok])

    return {
        "counts": {
            "total": len(outcomes),
            "analyzed": analyzed,
            **{k: v for k, v in status_counts.items() if k != "ok"},
        },
        "four_season": {
            "accuracy_top1": _accuracy(acc4_1),
            "accuracy_top2": _accuracy(acc4_2),
            "per_class": _per_class_accuracy(truths_four, preds_four),
            "confusion": _confusion(FOUR_SEASONS, truths_four, preds_four),
        },
        "twelve_season": {
            "accuracy_top1": _accuracy(acc12_1),
            "accuracy_top2": _accuracy(acc12_2),
            "accuracy_top3": _accuracy(acc12_3),
            "per_class": _per_class_accuracy(truths_twelve, preds_twelve),
            "confusion": _confusion(TWELVE_SEASONS, truths_twelve, preds_twelve),
        },
        "pipeline": {
            "parsing_used_rate": parsing_rate,
            "mean_confidence_four": round(
                float(np.mean([o.confidence_four or 0.0 for o in ok])), 4
            )
            if ok
            else None,
            "mean_confidence_twelve": round(
                float(np.mean([o.confidence_twelve or 0.0 for o in ok])), 4
            )
            if ok
            else None,
        },
    }


def _write_mistakes(outcomes: list[SampleOutcome], path: Path, limit: int = 200) -> None:
    mistakes = [
        {
            "path": str(o.sample.path),
            "truth_four": o.sample.season_four,
            "truth_twelve": o.sample.season_twelve,
            "pred_four": o.pred_four,
            "pred_twelve": o.pred_twelve,
            "top_four": o.top_four,
            "top_twelve": o.top_twelve,
        }
        for o in outcomes
        if o.status == "ok"
        and (o.pred_four != o.sample.season_four or o.pred_twelve != o.sample.season_twelve)
    ][:limit]
    path.write_text(json.dumps(mistakes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark seasonal analysis on Deep Armocromia RGB")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Path to RGB/ (auto-detect by default)")
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--limit", type=int, default=None, help="Max images to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N samples")
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "artefacts" / "eval" / "deep_armocromia_latest.json",
    )
    parser.add_argument("--mistakes", type=Path, default=None, help="Optional JSON with misclassified samples")
    parser.add_argument("--parsing-backend", type=str, default=None, help="Override FACE_AI_PARSING_BACKEND")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    dataset_root = resolve_dataset_root(args.dataset_root)
    samples = iter_samples(dataset_root, split=args.split)
    if args.offset:
        samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]

    if not samples:
        print("No samples found.", file=sys.stderr)
        return 1

    print(f"Dataset: {dataset_root}")
    print(f"Split: {args.split} | samples: {len(samples)}")
    print(
        f"Pipeline: parsing={args.parsing_backend or settings.parsing_backend}, "
        f"season={settings.season_classifier}, skip_photo_gate={settings.skip_analysis_if_photo_poor}"
    )

    t0 = time.perf_counter()
    outcomes: list[SampleOutcome] = []
    for i, sample in enumerate(samples, start=1):
        outcomes.append(analyze_sample(sample, parsing_backend=args.parsing_backend))
        if args.progress_every and i % args.progress_every == 0:
            ok = sum(1 for o in outcomes if o.status == "ok")
            print(f"  [{i}/{len(samples)}] analyzed={ok}")

    elapsed = time.perf_counter() - t0
    summary = aggregate(outcomes)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "split": args.split,
        "limit": args.limit,
        "offset": args.offset,
        "elapsed_sec": round(elapsed, 2),
        "settings": {
            "parsing_backend": args.parsing_backend or settings.parsing_backend,
            "season_classifier": settings.season_classifier,
            "skin_color_backend": settings.skin_color_backend,
            "lip_color_backend": settings.lip_color_backend,
            "landmark_backend": settings.landmark_backend,
        },
        **summary,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    mistakes_path = args.mistakes or args.output.with_name(args.output.stem + "_mistakes.json")
    _write_mistakes(outcomes, mistakes_path)

    c = summary["counts"]
    f4 = summary["four_season"]
    t12 = summary["twelve_season"]
    print()
    print(f"Done in {elapsed:.1f}s -> {args.output}")
    print(f"  analyzed: {c.get('analyzed', 0)}/{c['total']}  skipped: {c.get('no_face', 0)} no_face")
    if f4["accuracy_top1"] is not None:
        print(f"  4-season  top-1: {f4['accuracy_top1']:.1%}  top-2: {f4['accuracy_top2']:.1%}")
        print(
            f"  12-season top-1: {t12['accuracy_top1']:.1%}  "
            f"top-2: {t12['accuracy_top2']:.1%}  top-3: {t12['accuracy_top3']:.1%}"
        )
    print(f"  mistakes: {mistakes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
