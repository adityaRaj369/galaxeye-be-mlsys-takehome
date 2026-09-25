"""Offline tile service: ingest (single/batch), classify, store, query, review, canary."""

from __future__ import annotations

import csv
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.canary import load_canary, run_canary
from app.classifier import LandUseClassifier
from app.config import APP_LOG, CLASSES, EVAL_DIR, EVAL_LABELS, MODEL_VERSION, ROOT, VAR_DIR
from app.storage import PredictionStore, utc_now

classifier: LandUseClassifier | None = None
store: PredictionStore | None = None


class ReviewBody(BaseModel):
    corrected_label: str
    note: str = ""
    reviewer: str = Field(default="analyst")


def _log(line: str) -> None:
    VAR_DIR.mkdir(parents=True, exist_ok=True)
    with APP_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {line}\n")


@asynccontextmanager
async def lifespan(_: FastAPI):
    global classifier, store
    if classifier is None:
        classifier = LandUseClassifier()
    if store is None:
        store = PredictionStore()
    _log(f"startup model={classifier.version}")
    yield


app = FastAPI(
    title="GalaxEye tile classifier",
    description="Offline ingest, classify, store, query, review, and canary.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def _require_ready() -> tuple[LandUseClassifier, PredictionStore]:
    if classifier is None or store is None:
        raise HTTPException(status_code=503, detail="Service is still starting")
    return classifier, store


def _classify_and_store(clf: LandUseClassifier, db: PredictionStore, filename: str, data: bytes) -> dict:
    prediction = clf.predict_bytes(data)
    record = db.save(
        record_id=str(uuid.uuid4()),
        filename=filename,
        predicted_label=prediction.label,
        confidence=prediction.confidence,
        scores=prediction.scores,
        review_status=prediction.review_status,
        model_version=prediction.model_version,
        image_bytes=data,
    )
    _log(
        f"ingest id={record['id']} file={filename} pred={prediction.label} "
        f"conf={prediction.confidence} status={prediction.review_status}"
    )
    return record


@app.get("/health")
def health() -> dict:
    clf, db = _require_ready()
    return {
        "status": "ok",
        "offline": True,
        "model_version": clf.version,
        "confidence_threshold": clf.threshold,
        "classes": clf.classes,
        "eval_accuracy_at_train": clf.eval_accuracy,
        "store": db.stats(),
        "canary": load_canary(),
        "log_path": str(APP_LOG),
    }


@app.post("/tiles")
async def ingest_tile(file: Annotated[UploadFile, File(...)]) -> JSONResponse:
    clf, db = _require_ready()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        record = _classify_and_store(clf, db, file.filename, data)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read tile: {exc}") from exc
    return JSONResponse(status_code=201, content=record)


@app.post("/tiles/batch")
async def ingest_batch(files: Annotated[list[UploadFile], File(...)]) -> dict:
    """Upload many tiles in one request (a strip / folder dropped in the UI)."""
    clf, db = _require_ready()
    if not files:
        raise HTTPException(status_code=400, detail="No files")
    stored = []
    errors = []
    for file in files:
        name = file.filename or "unnamed.png"
        data = await file.read()
        if not data:
            errors.append({"filename": name, "error": "empty"})
            continue
        try:
            stored.append(_classify_and_store(clf, db, name, data))
        except Exception as exc:
            errors.append({"filename": name, "error": str(exc)})
    return {"stored": len(stored), "failed": len(errors), "results": stored, "errors": errors}


@app.post("/ingest/eval-set")
def ingest_eval_set() -> dict:
    """Offline folder ingest: classify every PNG in the provided eval_set."""
    clf, db = _require_ready()
    paths = sorted(EVAL_DIR.glob("*.png"))
    if not paths:
        raise HTTPException(status_code=404, detail=f"No tiles in {EVAL_DIR}")
    stored = []
    errors = []
    for path in paths:
        try:
            stored.append(_classify_and_store(clf, db, path.name, path.read_bytes()))
        except Exception as exc:
            errors.append({"filename": path.name, "error": str(exc)})
    return {"stored": len(stored), "failed": len(errors), "errors": errors}


@app.get("/tiles")
def list_tiles(
    label: Annotated[str | None, Query()] = None,
    review_status: Annotated[str | None, Query()] = None,
    min_confidence: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    q: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> list[dict]:
    _, db = _require_ready()
    if label and label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown label. Use one of: {CLASSES}")
    allowed = {"accepted", "needs_review", "reviewed"}
    if review_status and review_status not in allowed:
        raise HTTPException(status_code=400, detail=f"review_status must be one of {sorted(allowed)}")
    return db.list(
        label=label,
        review_status=review_status,
        min_confidence=min_confidence,
        q=q,
        limit=limit,
    )


@app.get("/tiles/{tile_id}")
def get_tile(tile_id: str) -> dict:
    _, db = _require_ready()
    record = db.get(tile_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    return record


@app.get("/tiles/{tile_id}/image")
def get_tile_image(tile_id: str) -> FileResponse:
    _, db = _require_ready()
    record = db.get(tile_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    path = Path(record["tile_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image file missing")
    return FileResponse(path)


@app.patch("/tiles/{tile_id}/review")
def review_tile(tile_id: str, body: ReviewBody) -> dict:
    """Analyst write-back: keep the model prediction, store the human label too."""
    _, db = _require_ready()
    if body.corrected_label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Unknown label. Use one of: {CLASSES}")
    record = db.apply_review(tile_id, corrected_label=body.corrected_label, note=body.note)
    if record is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    _log(f"review id={tile_id} corrected={body.corrected_label} reviewer={body.reviewer}")
    return record


@app.post("/canary")
def canary_run() -> dict:
    clf, _ = _require_ready()
    report = run_canary(clf)
    _log(f"canary accuracy={report['accuracy']} healthy={report['healthy']}")
    return report


@app.get("/canary")
def canary_get() -> dict:
    report = load_canary()
    if report is None:
        raise HTTPException(status_code=404, detail="No canary has been run yet. POST /canary.")
    return report


def _eval_label_map() -> dict[str, str]:
    if not EVAL_LABELS.exists():
        return {}
    with EVAL_LABELS.open(newline="", encoding="utf-8") as handle:
        return {row["filename"]: row["true_label"] for row in csv.DictReader(handle)}


from app.web import router as pages_router

app.include_router(pages_router)
app.mount("/static", StaticFiles(directory=str(ROOT / "app" / "static")), name="static")


@app.get("/demo/labels")
def demo_labels() -> dict[str, str]:
    return _eval_label_map()


@app.get("/demo/samples")
def demo_samples() -> list[dict]:
    labels = _eval_label_map()
    picked: dict[str, str] = {}
    for filename, label in labels.items():
        if label not in picked:
            picked[label] = filename
        if len(picked) == len(CLASSES):
            break
    return [
        {
            "filename": filename,
            "expected_label": label,
            "url": f"/demo/tiles/{filename}",
        }
        for label, filename in picked.items()
    ]


@app.get("/demo/tiles/{filename}")
def demo_tile(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = EVAL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample tile not found")
    return FileResponse(path)
