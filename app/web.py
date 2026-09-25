"""HTML pages with real routes. JSON APIs stay in main.py."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from app.canary import load_canary, run_canary
from app.config import CLASSES, EVAL_DIR, ROOT
from app.storage import utc_now

router = APIRouter()
templates = Jinja2Templates(directory=str(ROOT / "app" / "templates"))

NAV = [
    ("/", "Overview"),
    ("/ingest", "Ingest"),
    ("/catalog", "Catalog"),
    ("/review", "Review"),
    ("/how-we-train", "How we train"),
    ("/how-we-predict", "How we predict"),
    ("/model", "Model"),
]


def _ready():
    from app.main import _eval_label_map, _require_ready

    return _require_ready(), _eval_label_map


def _ctx(request: Request, active: str, **extra) -> dict:
    (clf, db), _ = _ready()
    stats = db.stats()
    by_label = stats.get("by_label") or {}
    return {
        "request": request,
        "active": active,
        "nav": NAV,
        "classes": CLASSES,
        "model_version": clf.version,
        "threshold": clf.threshold,
        "eval_acc": clf.eval_accuracy,
        "stats": stats,
        "canary": load_canary(),
        "bar_max": max([1, *by_label.values()]),
        **extra,
    }


def _samples() -> list[dict]:
    (_, _), label_fn = _ready()
    labels = label_fn()
    picked: dict[str, str] = {}
    for filename, label in labels.items():
        if label not in picked:
            picked[label] = filename
        if len(picked) == len(CLASSES):
            break
    return [
        {"filename": name, "expected_label": label, "url": f"/demo/tiles/{name}"}
        for label, name in picked.items()
    ]


@router.get("/")
def overview(request: Request):
    return templates.TemplateResponse(request, "overview.html", _ctx(request, "/"))


@router.get("/ingest")
def ingest_page(request: Request):
    return templates.TemplateResponse(request, "ingest.html", _ctx(request, "/ingest", samples=_samples()))


@router.post("/ingest")
async def ingest_form(file: UploadFile = File(...)):
    from app.main import _classify_and_store, _require_ready

    clf, db = _require_ready()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    record = _classify_and_store(clf, db, file.filename, data)
    return RedirectResponse(f"/review/{record['id']}", status_code=303)


@router.post("/ingest/sample/{filename}")
def ingest_sample(filename: str):
    from app.main import _classify_and_store, _require_ready

    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = EVAL_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Sample not found")
    clf, db = _require_ready()
    record = _classify_and_store(clf, db, filename, path.read_bytes())
    return RedirectResponse(f"/review/{record['id']}", status_code=303)


@router.post("/ingest/eval-folder")
def ingest_eval_folder():
    from app.main import _classify_and_store, _require_ready

    clf, db = _require_ready()
    paths = sorted(EVAL_DIR.glob("*.png"))
    if not paths:
        raise HTTPException(status_code=404, detail="No eval tiles")
    for path in paths:
        _classify_and_store(clf, db, path.name, path.read_bytes())
    return RedirectResponse("/catalog?flash=ingested-eval-set", status_code=303)


@router.get("/catalog")
def catalog_page(
    request: Request,
    label: str | None = None,
    review_status: str | None = None,
    q: str | None = None,
    flash: str | None = None,
):
    from app.main import _require_ready

    _, db = _require_ready()
    tiles = db.list(label=label or None, review_status=review_status or None, q=q or None, limit=200)
    msg = "Ingested eval_set. Showing stored rows." if flash == "ingested-eval-set" else None
    return templates.TemplateResponse(
        request,
        "catalog.html",
        _ctx(
            request,
            "/catalog",
            tiles=tiles,
            label=label or "",
            review_status=review_status or "",
            q=q or "",
            flash=msg,
        ),
    )


@router.get("/review")
def review_list(request: Request):
    from app.main import _require_ready

    _, db = _require_ready()
    tiles = db.list(review_status="needs_review", limit=200)
    return templates.TemplateResponse(request, "review_list.html", _ctx(request, "/review", tiles=tiles))


@router.get("/review/{tile_id}")
def review_detail(request: Request, tile_id: str, saved: int | None = None):
    from app.main import _eval_label_map, _require_ready

    _, db = _require_ready()
    tile = db.get(tile_id)
    if tile is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    expected = _eval_label_map().get(tile["filename"])
    scores = sorted((tile.get("scores") or {}).items(), key=lambda kv: kv[1], reverse=True)
    tile = {**tile, "scores": dict(scores)}
    return templates.TemplateResponse(
        request,
        "review_detail.html",
        _ctx(
            request,
            "/review",
            tile=tile,
            expected=expected,
            flash="Correction saved." if saved else None,
        ),
    )


@router.post("/review/{tile_id}")
def review_form(tile_id: str, corrected_label: str = Form(...), note: str = Form("")):
    from app.main import _log, _require_ready

    _, db = _require_ready()
    if corrected_label not in CLASSES:
        raise HTTPException(status_code=400, detail="Unknown label")
    record = db.apply_review(tile_id, corrected_label=corrected_label, note=note)
    if record is None:
        raise HTTPException(status_code=404, detail="Tile not found")
    _log(f"review id={tile_id} corrected={corrected_label} reviewer=analyst")
    return RedirectResponse(f"/review/{tile_id}?saved=1", status_code=303)


@router.get("/how-we-train")
def train_page(request: Request):
    return templates.TemplateResponse(request, "train.html", _ctx(request, "/how-we-train"))


@router.get("/how-we-predict")
def predict_page(request: Request):
    return templates.TemplateResponse(request, "predict.html", _ctx(request, "/how-we-predict"))


@router.get("/model")
def model_page(request: Request):
    return templates.TemplateResponse(request, "model.html", _ctx(request, "/model"))


@router.post("/model/canary")
def model_canary():
    from app.main import _log, _require_ready

    clf, _ = _require_ready()
    report = run_canary(clf)
    _log(f"canary accuracy={report['accuracy']} healthy={report['healthy']} at={utc_now()}")
    return RedirectResponse("/model", status_code=303)
