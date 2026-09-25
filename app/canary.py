"""Offline health canary: re-score the frozen eval gold set and write a local file."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from app.classifier import LandUseClassifier
from app.config import CANARY_PATH, CLASSES, EVAL_DIR, EVAL_LABELS
from app.storage import utc_now


def run_canary(clf: LandUseClassifier, out_path: Path = CANARY_PATH) -> dict:
    truth = {}
    with EVAL_LABELS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            truth[row["filename"]] = row["true_label"]

    correct = 0
    total = 0
    by_class_correct: Counter[str] = Counter()
    by_class_total: Counter[str] = Counter()
    mistakes: Counter[str] = Counter()
    confusion = {src: {dst: 0 for dst in CLASSES} for src in CLASSES}

    for filename, expected in truth.items():
        path = EVAL_DIR / filename
        if not path.is_file():
            continue
        pred = clf.predict_bytes(path.read_bytes())
        total += 1
        by_class_total[expected] += 1
        if expected in confusion and pred.label in confusion[expected]:
            confusion[expected][pred.label] += 1
        if pred.label == expected:
            correct += 1
            by_class_correct[expected] += 1
        else:
            mistakes[f"{expected} → {pred.label}"] += 1

    baseline = clf.eval_accuracy
    accuracy = round(correct / total, 4) if total else 0.0
    drop = None if baseline is None else round(float(baseline) - accuracy, 4)
    healthy = True
    if baseline is not None and drop is not None and drop > 0.08:
        healthy = False

    report = {
        "ran_at": utc_now(),
        "model_version": clf.version,
        "tiles_scored": total,
        "accuracy": accuracy,
        "baseline_accuracy_at_train": baseline,
        "accuracy_drop_vs_train": drop,
        "healthy": healthy,
        "per_class_accuracy": {
            label: round(by_class_correct[label] / by_class_total[label], 4)
            if by_class_total[label]
            else None
            for label in CLASSES
        },
        "top_mistakes": mistakes.most_common(8),
        "confusion_matrix": confusion,
        "note": "This canary never talks to the network. POST /canary or use the Model page.",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_canary(out_path: Path = CANARY_PATH) -> dict | None:
    if not out_path.exists():
        return None
    return json.loads(out_path.read_text(encoding="utf-8"))
