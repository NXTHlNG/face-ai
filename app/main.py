from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.exceptions import DlibNotInstalledError
from app.schemas.analysis import AnalysisResponse
from app.services.analysis_service import analyze_image_bytes

app = FastAPI(
    title="Face style analysis",
    version="0.1.0",
    description="Core API: метрики лица и эвристические рекомендации по стилю.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...)) -> AnalysisResponse:
    data = await file.read()
    try:
        return analyze_image_bytes(data)
    except DlibNotInstalledError as e:
        raise HTTPException(
            status_code=503,
            detail={"error": "dlib_required", "message": str(e)},
        ) from e


@app.get("/schema.json")
def analysis_json_schema() -> JSONResponse:
    path = Path(__file__).resolve().parent / "schemas" / "analysis_contract.json"
    if path.is_file():
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    schema = AnalysisResponse.model_json_schema()
    return JSONResponse(schema)


def create_app() -> FastAPI:
    return app
