"""Train the local classifier on candidate_tiles and measure it on eval_set."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import (  # noqa: E402
    ARTIFACT_DIR,
    CANDIDATE_DIR,
    CLASSES,
    CONFIDENCE_THRESHOLD,
    EVAL_DIR,
    EVAL_LABELS,
    MODEL_PATH,
    MODEL_VERSION,
)
from app.features import extract_features, load_rgb_tile  # noqa: E402


def load_folder_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    labels = []
    for label in CLASSES:
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")
        for path in sorted(class_dir.glob("*.png")):
            vectors.append(extract_features(load_rgb_tile(path.read_bytes())))
            labels.append(label)
    return np.vstack(vectors), np.array(labels)


def load_eval_set() -> tuple[np.ndarray, np.ndarray, list[str]]:
    truth = {}
    with EVAL_LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[row["filename"]] = row["true_label"]

    vectors = []
    labels = []
    names = []
    for path in sorted(EVAL_DIR.glob("*.png")):
        if path.name not in truth:
            continue
        vectors.append(extract_features(load_rgb_tile(path.read_bytes())))
        labels.append(truth[path.name])
        names.append(path.name)
    return np.vstack(vectors), np.array(labels), names


def main() -> None:
    print(f"Loading candidate tiles from {CANDIDATE_DIR}")
    X, y = load_folder_dataset(CANDIDATE_DIR)
    print(f"Training tiles: {len(y)}  class counts: {dict(Counter(y))}")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    holdout_acc = float(model.score(X_holdout, y_holdout))
    print(f"Holdout accuracy on candidate_tiles: {holdout_acc:.3f}")

    # Retrain on all labelled candidate tiles before locking the artifact.
    model.fit(X, y)

    print(f"Evaluating on {EVAL_DIR}")
    X_eval, y_eval, _ = load_eval_set()
    y_pred = model.predict(X_eval)
    eval_acc = float((y_pred == y_eval).mean())
    print(f"Eval accuracy: {eval_acc:.3f}")
    print(classification_report(y_eval, y_pred, labels=CLASSES, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_eval, y_pred, labels=CLASSES))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "classes": CLASSES,
        "version": MODEL_VERSION,
        "threshold": CONFIDENCE_THRESHOLD,
        "eval_accuracy": round(eval_acc, 4),
        "holdout_accuracy": round(holdout_acc, 4),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Wrote {MODEL_PATH}")


if __name__ == "__main__":
    main()
