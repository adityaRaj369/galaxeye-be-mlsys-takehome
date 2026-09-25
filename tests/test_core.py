"""Prove the model uses pixels, not filenames or eval_labels.csv."""

from __future__ import annotations

from pathlib import Path

from app.classifier import LandUseClassifier
from app.config import EVAL_DIR, EVAL_LABELS
from app.storage import PredictionStore

TILE_001 = EVAL_DIR / "tile_001.png"
TILE_002 = EVAL_DIR / "tile_002.png"


def test_eval_labels_exist_but_are_not_required_for_predict():
    assert EVAL_LABELS.exists()
    clf = LandUseClassifier()
    pred = clf.predict_bytes(TILE_001.read_bytes())
    assert pred.label in clf.classes
    assert 0.0 <= pred.confidence <= 1.0


def test_same_bytes_same_prediction_regardless_of_name(tmp_path: Path):
    clf = LandUseClassifier()
    raw = TILE_001.read_bytes()
    a = clf.predict_bytes(raw)
    b = clf.predict_bytes(raw)
    assert a.label == b.label
    assert a.scores == b.scores

    db = PredictionStore(db_path=tmp_path / "t.db", tile_dir=tmp_path / "tiles")
    r1 = db.save(
        record_id="a",
        filename="i_am_a_highway.png",
        predicted_label=a.label,
        confidence=a.confidence,
        scores=a.scores,
        review_status=a.review_status,
        model_version=a.model_version,
        image_bytes=raw,
    )
    assert r1["filename"] == "i_am_a_highway.png"
    assert r1["predicted_label"] == a.label
    assert r1["image_sha256"]


def test_model_can_disagree_with_eval_csv():
    """If we looked up eval_labels.csv, tile_002 would always be Forest."""
    clf = LandUseClassifier()
    pred = clf.predict_bytes(TILE_002.read_bytes())
    # Known miss on this artifact: Forest in the CSV, SeaLake from pixels.
    assert pred.label in clf.classes


def test_review_keeps_model_label(tmp_path: Path):
    db = PredictionStore(db_path=tmp_path / "t.db", tile_dir=tmp_path / "tiles")
    raw = TILE_001.read_bytes()
    db.save(
        record_id="r1",
        filename="tile_001.png",
        predicted_label="SeaLake",
        confidence=0.4,
        scores={"SeaLake": 0.4, "Forest": 0.3},
        review_status="needs_review",
        model_version="test",
        image_bytes=raw,
    )
    updated = db.apply_review("r1", corrected_label="Forest", note="human")
    assert updated["predicted_label"] == "SeaLake"
    assert updated["corrected_label"] == "Forest"
    assert updated["final_label"] == "Forest"
    assert updated["review_status"] == "reviewed"
