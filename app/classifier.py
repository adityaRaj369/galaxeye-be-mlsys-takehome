"""Local land-use classifier. No network calls at inference time."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from app.config import CLASSES, CONFIDENCE_THRESHOLD, MODEL_PATH, MODEL_VERSION
from app.features import features_from_bytes, load_rgb_tile


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    scores: dict[str, float]
    review_status: str
    model_version: str


class LandUseClassifier:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Run: python scripts/train.py"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        # predict_proba columns follow sklearn's classes_, not our config list.
        self.classes = list(getattr(self.model, "classes_", bundle.get("classes", CLASSES)))
        self.version = bundle.get("version", MODEL_VERSION)
        self.threshold = float(bundle.get("threshold", CONFIDENCE_THRESHOLD))
        self.eval_accuracy = bundle.get("eval_accuracy")

    def predict_bytes(self, data: bytes) -> Prediction:
        # Load once so a corrupt file fails before we touch the model.
        load_rgb_tile(data)
        vector = features_from_bytes(data).reshape(1, -1)
        probabilities = self.model.predict_proba(vector)[0]
        scores = {
            label: float(prob)
            for label, prob in zip(self.classes, probabilities)
        }
        best_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[best_idx])
        return Prediction(
            label=self.classes[best_idx],
            confidence=round(confidence, 4),
            scores={k: round(v, 4) for k, v in scores.items()},
            review_status="accepted" if confidence >= self.threshold else "needs_review",
            model_version=self.version,
        )
