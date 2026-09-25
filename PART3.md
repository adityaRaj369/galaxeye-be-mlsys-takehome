# Part 3 — problem-solving

These are the four scenario questions from the GalaxEye brief.
Short answers. How I reason, not a product spec.

---

## 1. The classifier is wrong about 30% of the time. What do you do — and how do you decide if it is “good enough”?

I would not retrain first. I would ask **which errors, for which class, and what they cost**.

“30% wrong” is an average. On this zip I would use their `eval_set` + `eval_labels.csv`: confusion matrix, per-class precision/recall, and accuracy by confidence bucket. If almost all mistakes are Highway ↔ Industrial and Forest is clean, the service may already be useful for a forest-cover question and useless for infrastructure.

“Good enough” is a product test:

- What decision does the analyst make with a label?
- What is the cost of a false Industrial vs a missed River?
- After we hold out low-confidence tiles for review, what is the error rate on the **auto-accepted** slice?

If auto-accepted error is low and a human can clear the review pile, 70% raw accuracy can still be useful. If every class is noisy and confidence is uncalibrated, it is only a sort hint, not automation.

What I would do: keep storing full scores, raise the threshold so the machine only auto-accepts when it is usually right, send the rest to review, and use those corrections as the next training set. A new backbone comes after I know which confusion I am paying for.

---

## 2. The service runs offline, with no one watching it live. A month later, how would you know it is still working correctly?

I would not rely on a cloud dashboard. The box must **write evidence on disk**.

1. Heartbeat: process up, model file hash, last successful inference time, disk free, SQLite `integrity_check`.
2. **Canary:** re-score a frozen gold set (their eval set) on a schedule. Write accuracy and the confusion matrix next to the deploy-time baseline. If accuracy drops or one class collapses, the box is sick even if `/tiles` still returns 200.
3. Local structured logs: request id, model version, latency, label, confidence. Rotate them. A month of silence with no canary is not “healthy”.
4. A watchdog that restarts a dead process and leaves a crash breadcrumb.

In this repo, `GET /health` and `POST /canary` (also the Model page) are that start. The failure I care about offline is **silent wrong**, not crash.

---

## 3. Tiles are coming in fine, but the stored results look wrong. Walk through how you find the cause — steps in order.

I split it into **wrong image, wrong model, wrong write**.

1. Pick one bad id in SQLite. Read filename, predicted_label, scores, model_version, tile_path, created_at.
2. Open the stored file at `tile_path`. Is it the tile the analyst thinks it is? If not, ingest mix-up. Stop. Do not touch the model.
3. Re-run `LandUseClassifier` on those exact bytes. Same label and scores? If no: two artifacts or preprocess drift (resize, RGB vs RGBA).
4. Check `model_version` against the joblib we think is deployed. Hash the file.
5. Look at the full score vector. Confident-and-wrong vs 0.34 / 0.33 / 0.31 is a different bug (threshold / review, not a corrupt DB).
6. Run the eval set through the same binary. If eval is still at the train-time number, the model is fine and the new tiles are out-of-distribution (or the expected label is wrong). If eval also collapsed, the code or artifact changed.
7. Only then look at training: class mapping, folder names, label file.
8. Last: storage. `PRAGMA integrity_check`, wrong row, or reading a different DB file than we write.

I would not start by tweaking hyperparameters. Most “results look wrong” bugs here are path, version, or expectation.

---

## 4. What is the weakest part of your design, and what would break it first?

The weakest part is the **model, not the API**. Colour histograms and an 8×8 thumbnail will not survive a sensor change, a new GSD, heavy cloud, or a class defined by texture we threw away (Highway vs Industrial).

What breaks first: **distribution shift**. New satellite, 256×256 chips, or a new geography — confidence stays high and labels go bad. The review flag will not save confidently wrong predictions.

Second: one global threshold. We mark `needs_review`; a human can write a corrected label, but we do not yet retrain from that loop automatically.

Third: SQLite + one process. Fine for this assignment. The first ops break is concurrent ingest or a full disk (we write every PNG again).
