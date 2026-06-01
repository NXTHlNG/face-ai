"""Smoke test for Deep Armocromia seasonal benchmark (skipped if dataset missing)."""

from pathlib import Path

import pytest

from app.eval.deep_armocromia import iter_samples, resolve_dataset_root


def _dataset_available() -> bool:
    try:
        resolve_dataset_root()
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _dataset_available(), reason="Deep Armocromia dataset not present")
def test_dataset_has_test_split():
    samples = iter_samples(split="test")
    assert len(samples) >= 100
    assert all(s.season_four in {"spring", "summer", "autumn", "winter"} for s in samples[:20])


@pytest.mark.skipif(not _dataset_available(), reason="Deep Armocromia dataset not present")
def test_eval_script_runs_on_two_images():
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    out = root / "artefacts" / "eval" / "_pytest_smoke.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "eval_season_dataset.py"),
            "--split",
            "test",
            "--limit",
            "2",
            "--output",
            str(out),
            "--progress-every",
            "0",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.is_file()
    data = out.read_text(encoding="utf-8")
    assert "four_season" in data
    assert "twelve_season" in data
