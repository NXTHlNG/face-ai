from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.exceptions import DlibNotInstalledError
from app.pipeline.orchestrator import health_status
from app.schemas.analysis import AnalysisRequestMeta, AnalysisResponse
from app.schemas.products import ProductCategory, ProductMatchResponse
from app.services.analysis_service import analyze_image_bytes
from app.services.product_catalog import ProductCatalogError, catalog_stats
from app.services.product_matcher import match_products


def _configure_app_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    for name in ("app.backends.llm", "app.pipeline.orchestrator"):
        log = logging.getLogger(name)
        log.handlers.clear()
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


_configure_app_logging()

app = FastAPI(
    title="Face style analysis",
    version="0.2.0",
    description="Seasonal color analysis: parsing, Munsell axes, ensemble 4/12/16 seasons.",
)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
_OVERLAYS_DIR = _STATIC_DIR / "overlays"
if _OVERLAYS_DIR.is_dir():
    app.mount("/static/overlays", StaticFiles(directory=str(_OVERLAYS_DIR)), name="overlays")


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Fast liveness probe — must not load ML backends (SegFace/torch)."""
    return {
        "status": "ok",
        "rules_version": settings.rules_version,
        "llm_available": settings.llm_available,
        "generative_available": settings.generative_available,
        "products_catalog": catalog_stats(),
    }


@app.get("/health/full")
def health_full() -> dict:
    """Detailed health including parsing backend checks (may take a few seconds)."""
    return health_status()


def _resolve_debug_path(run_id: str, filename: str) -> Path:
    if ".." in run_id or ".." in filename:
        raise HTTPException(status_code=400, detail="invalid path")
    if "/" in run_id or "\\" in run_id or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="invalid path")
    base = settings.debug_output_dir.resolve()
    path = (base / run_id / filename).resolve()
    if not str(path).startswith(str(base)) or not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return path


@app.get("/debug/{run_id}/{filename}")
def debug_artifact(run_id: str, filename: str) -> FileResponse:
    return FileResponse(_resolve_debug_path(run_id, filename))


@app.post("/analyze", response_model=AnalysisResponse)
@app.post("/analyse", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    wrist_file: UploadFile | None = File(None),
    meta_json: Annotated[str | None, Form()] = None,
) -> AnalysisResponse:
    data = await file.read()
    wrist_data = await wrist_file.read() if wrist_file else None
    meta = None
    if meta_json:
        try:
            meta = AnalysisRequestMeta.model_validate_json(meta_json)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"invalid meta_json: {e}") from e
    try:
        return await run_in_threadpool(
            analyze_image_bytes,
            data,
            wrist_data=wrist_data,
            meta=meta,
        )
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
    return JSONResponse(AnalysisResponse.model_json_schema())


@app.get("/products/match", response_model=ProductMatchResponse)
def products_match(
    season_twelve: str = Query(..., min_length=3),
    category: ProductCategory = Query(...),
    top_k: int = Query(3, ge=1, le=20),
    target_l: float | None = Query(None),
    target_a: float | None = Query(None),
    target_b: float | None = Query(None),
) -> ProductMatchResponse:
    target_lab = None
    if target_l is not None and target_a is not None and target_b is not None:
        target_lab = (target_l, target_a, target_b)
    try:
        return match_products(
            season_twelve,
            category,
            target_lab=target_lab,
            top_k=top_k,
        )
    except ProductCatalogError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def create_app() -> FastAPI:
    return app
