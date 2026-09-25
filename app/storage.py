"""SQLite-backed prediction store. Queryable without a database server."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from app.config import DB_PATH, TILE_STORE_DIR, VAR_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class PredictionStore:
    def __init__(self, db_path: Path = DB_PATH, tile_dir: Path | None = None) -> None:
        VAR_DIR.mkdir(parents=True, exist_ok=True)
        self.tile_dir = Path(tile_dir) if tile_dir else TILE_STORE_DIR
        self.tile_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    scores_json TEXT NOT NULL,
                    review_status TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    tile_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            existing = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
            extras = {
                "corrected_label": "TEXT",
                "reviewed_at": "TEXT",
                "review_note": "TEXT",
                "image_sha256": "TEXT",
            }
            for name, typ in extras.items():
                if name not in existing:
                    conn.execute(f"ALTER TABLE predictions ADD COLUMN {name} {typ}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_label ON predictions(predicted_label)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_status ON predictions(review_status)"
            )

    def save(
        self,
        *,
        record_id: str,
        filename: str,
        predicted_label: str,
        confidence: float,
        scores: dict[str, float],
        review_status: str,
        model_version: str,
        image_bytes: bytes,
    ) -> dict:
        tile_path = self.tile_dir / f"{record_id}.png"
        tile_path.write_bytes(image_bytes)
        digest = hashlib.sha256(image_bytes).hexdigest()
        created_at = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    id, filename, predicted_label, confidence, scores_json,
                    review_status, model_version, tile_path, created_at, image_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id,
                    filename,
                    predicted_label,
                    confidence,
                    json.dumps(scores),
                    review_status,
                    model_version,
                    str(tile_path),
                    created_at,
                    digest,
                ),
            )
        return self.get(record_id)

    def apply_review(self, record_id: str, *, corrected_label: str, note: str = "") -> dict | None:
        if self.get(record_id) is None:
            return None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE predictions
                SET corrected_label = ?,
                    review_note = ?,
                    reviewed_at = ?,
                    review_status = 'reviewed'
                WHERE id = ?
                """,
                (corrected_label, note, utc_now(), record_id),
            )
        return self.get(record_id)

    def get(self, record_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE id = ?", (record_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(
        self,
        *,
        label: str | None = None,
        review_status: str | None = None,
        min_confidence: float | None = None,
        q: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        clauses = []
        params: list[object] = []
        if label:
            clauses.append("predicted_label = ?")
            params.append(label)
        if review_status:
            clauses.append("review_status = ?")
            params.append(review_status)
        if min_confidence is not None:
            clauses.append("confidence >= ?")
            params.append(min_confidence)
        if q:
            clauses.append("filename LIKE ?")
            params.append(f"%{q}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"""
            SELECT * FROM predictions
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]
            last = conn.execute(
                "SELECT created_at FROM predictions ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            by_label = conn.execute(
                """
                SELECT predicted_label, COUNT(*) AS n
                FROM predictions
                GROUP BY predicted_label
                ORDER BY n DESC
                """
            ).fetchall()
            by_status = conn.execute(
                """
                SELECT review_status, COUNT(*) AS n
                FROM predictions
                GROUP BY review_status
                """
            ).fetchall()
        return {
            "tiles_stored": total,
            "last_inference_at": last["created_at"] if last else None,
            "by_label": {row["predicted_label"]: row["n"] for row in by_label},
            "by_review_status": {row["review_status"]: row["n"] for row in by_status},
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        keys = row.keys()
        corrected = row["corrected_label"] if "corrected_label" in keys else None
        predicted = row["predicted_label"]
        return {
            "id": row["id"],
            "filename": row["filename"],
            "predicted_label": predicted,
            "corrected_label": corrected,
            "final_label": corrected or predicted,
            "confidence": row["confidence"],
            "scores": json.loads(row["scores_json"]),
            "review_status": row["review_status"],
            "review_note": row["review_note"] if "review_note" in keys else None,
            "reviewed_at": row["reviewed_at"] if "reviewed_at" in keys else None,
            "model_version": row["model_version"],
            "tile_path": row["tile_path"],
            "image_sha256": row["image_sha256"] if "image_sha256" in keys else None,
            "image_url": f"/tiles/{row['id']}/image",
            "created_at": row["created_at"],
        }
