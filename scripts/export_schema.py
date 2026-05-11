"""Writes app/schemas/analysis_contract.json from Pydantic models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.schemas.analysis import AnalysisResponse  # noqa: E402


def main() -> None:
    path = ROOT / "app" / "schemas" / "analysis_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = AnalysisResponse.model_json_schema()
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
