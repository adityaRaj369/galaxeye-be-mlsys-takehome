from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.classifier import LandUseClassifier
from app.config import EVAL_DIR
from app.storage import PredictionStore

TILE = EVAL_DIR / "tile_001.png"


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import app.main as main

    main.classifier = LandUseClassifier()
    main.store = PredictionStore(db_path=tmp_path / "api.db", tile_dir=tmp_path / "tiles")
    with TestClient(main.app) as test_client:
        yield test_client
    main.classifier = None
    main.store = None


def test_health(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["offline"] is True
    assert body["model_version"]


@pytest.mark.parametrize(
    "path",
    ["/", "/ingest", "/catalog", "/review", "/how-we-train", "/how-we-predict", "/model"],
)
def test_html_routes(client: TestClient, path: str):
    res = client.get(path)
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_ingest_classify_store_and_fetch(client: TestClient):
    res = client.post("/tiles", files={"file": ("renamed.png", TILE.read_bytes(), "image/png")})
    assert res.status_code == 201
    body = res.json()
    assert body["filename"] == "renamed.png"
    assert body["predicted_label"] == "Forest"
    assert body["image_sha256"]
    fetched = client.get(f"/tiles/{body['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == body["id"]
    img = client.get(f"/tiles/{body['id']}/image")
    assert img.status_code == 200


def test_empty_upload_rejected(client: TestClient):
    res = client.post("/tiles", files={"file": ("empty.png", b"", "image/png")})
    assert res.status_code == 400


def test_review_endpoint(client: TestClient):
    created = client.post("/tiles", files={"file": ("t.png", TILE.read_bytes(), "image/png")})
    tile_id = created.json()["id"]
    res = client.patch(
        f"/tiles/{tile_id}/review",
        json={"corrected_label": "River", "note": "test"},
    )
    assert res.status_code == 200
    assert res.json()["final_label"] == "River"
    assert res.json()["predicted_label"] == created.json()["predicted_label"]
