from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

SEASON_IT_TO_EN: dict[str, str] = {
    "primavera": "spring",
    "estate": "summer",
    "autunno": "autumn",
    "inverno": "winter",
}

# Deep Armocromia Flow Theory folders → face-ai 12-season ids.
SUBTYPE_TO_TWELVE: dict[tuple[str, str], str] = {
    ("primavera", "warm"): "true_spring",
    ("primavera", "light"): "light_spring",
    ("primavera", "bright"): "bright_spring",
    ("estate", "cool"): "true_summer",
    ("estate", "light"): "light_summer",
    ("estate", "soft"): "soft_summer",
    ("autunno", "warm"): "true_autumn",
    ("autunno", "soft"): "soft_autumn",
    ("autunno", "deep"): "deep_autumn",
    ("inverno", "cool"): "true_winter",
    ("inverno", "bright"): "bright_winter",
    ("inverno", "deep"): "deep_winter",
}

TWELVE_SEASONS = sorted({v for v in SUBTYPE_TO_TWELVE.values()})
FOUR_SEASONS = sorted(set(SEASON_IT_TO_EN.values()))


@dataclass(frozen=True)
class DatasetSample:
    path: Path
    season_it: str
    subtype_it: str
    season_four: str
    season_twelve: str
    split: str


def resolve_dataset_root(root: Path | None = None) -> Path:
    """Locate Deep Armocromia RGB tree under ``dataset/``."""
    candidates: list[Path] = []
    if root is not None:
        candidates.append(root)
    project_root = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            project_root / "dataset" / "RGB" / "RGB",
            project_root / "dataset" / "RGB",
            project_root / "dataset",
        ]
    )
    for base in candidates:
        if not base.exists():
            continue
        for split in ("test", "train"):
            if (base / split / "primavera").exists():
                return base
        nested = base / "RGB"
        if (nested / "test" / "primavera").exists():
            return nested
    raise FileNotFoundError(
        "Deep Armocromia dataset not found. Expected dataset/RGB/RGB/{train,test}/<season>/..."
    )


def _map_labels(season_it: str, subtype_it: str) -> tuple[str, str]:
    season_four = SEASON_IT_TO_EN.get(season_it.lower())
    if season_four is None:
        raise ValueError(f"Unknown season folder: {season_it!r}")
    key = (season_it.lower(), subtype_it.lower())
    season_twelve = SUBTYPE_TO_TWELVE.get(key)
    if season_twelve is None:
        raise ValueError(f"Unknown subtype folder {subtype_it!r} under {season_it!r}")
    return season_four, season_twelve


def iter_samples(
    dataset_root: Path | None = None,
    *,
    split: str = "test",
) -> list[DatasetSample]:
    base = resolve_dataset_root(dataset_root)
    splits = ["test", "train"] if split == "all" else [split]
    samples: list[DatasetSample] = []
    for sp in splits:
        split_dir = base / sp
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split not found: {split_dir}")
        for season_dir in sorted(split_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            for subtype_dir in sorted(season_dir.iterdir()):
                if not subtype_dir.is_dir():
                    continue
                season_four, season_twelve = _map_labels(season_dir.name, subtype_dir.name)
                for img in sorted(subtype_dir.iterdir()):
                    if img.suffix.lower() not in IMAGE_SUFFIXES:
                        continue
                    samples.append(
                        DatasetSample(
                            path=img,
                            season_it=season_dir.name,
                            subtype_it=subtype_dir.name,
                            season_four=season_four,
                            season_twelve=season_twelve,
                            split=sp,
                        )
                    )
    return samples
