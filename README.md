# GalaxEye take-home — Backend Engineer, ML Systems

Offline slice of a satellite-tile land-use service.

- **Part 1:** [DESIGN.pdf](DESIGN.pdf) and the picture [DESIGN.png](DESIGN.png). Text: [DESIGN.md](DESIGN.md)
- **Part 2:** this repo (core path: `POST /tiles`)
- **Part 3:** the four scenario answers — **[PART3.md](PART3.md)** (also copied at the bottom of this file)

### Submit (by 27 Sept 2026 EOD)

Open their form: https://forms.cloud.microsoft/r/19966HsT0C

Paste the GitHub repo URL. That repo is this folder — `DESIGN.pdf`, `PART3.md`, this `README.md`, `app/`, `scripts/train.py`, `artifacts/landuse_clf.joblib`, `requirements.txt`, `tests/`, and `data/`. Do not upload `.venv/` or `var/`.

```powershell
pytest
```

That is the automated proof the core path works.

## How to test (do this)

The model is not a ChatGPT popup. It is a file:

`artifacts/landuse_clf.joblib`

That file is the trained classifier. The app loads it and uses it on every upload.

1. Start the API:

```powershell
cd C:\Users\anura\galaxyeye
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

2. Open the desk in a browser. These are real page routes:

```text
http://127.0.0.1:8000/                 Overview
http://127.0.0.1:8000/ingest           Upload / sample tiles
http://127.0.0.1:8000/catalog          Query stored results
http://127.0.0.1:8000/review           Low-confidence queue
http://127.0.0.1:8000/review/{id}      One tile + correction form
http://127.0.0.1:8000/how-we-train     How training used their folders
http://127.0.0.1:8000/how-we-predict   How POST /tiles classifies pixels
http://127.0.0.1:8000/model            Model card + canary matrix
```

   Click one of the 7 sample tiles (each has a known expected label), or
   upload a PNG from `data/be-mlsys-assignment-dataset/eval_set/`.

3. You should see: **expected label**, **model prediction**, **match yes/no**,
   **confidence**, **review status**, and the **stored id**.

4. Model health: open **http://127.0.0.1:8000/model** and run the canary.

## What results are expected

GalaxEye **did not set a target accuracy**. The PDF says they are not grading
accuracy and a model that is often wrong is fine.

A **correct system result** for one tile looks like this:

```json
{
  "id": "…",
  "filename": "tile_001.png",
  "predicted_label": "Forest",
  "confidence": 0.99,
  "review_status": "accepted",
  "scores": { "Forest": 0.99, "SeaLake": 0.01, "…" : 0 }
}
```

Meaning:

| Field | What “good” means |
|---|---|
| `predicted_label` | One of the 7 classes, always |
| `confidence` | 0 to 1. Below 0.50 → `needs_review` |
| `review_status` | `accepted` or `needs_review` |
| stored in SQLite | You can `GET /tiles/{id}` and get the same row back |

On **this** dataset, after training, the model is about **77%** vs
`eval_labels.csv`. So out of 210 eval tiles, expect ~160 matches and
~50 misses. Highway is the weakest class. That is a valid outcome.

Easy check: `tile_001.png` expected **Forest**. The model usually predicts
Forest with high confidence.

## What this slice does

One working path:

1. Accept a satellite tile (`POST /tiles`)
2. Classify it with a **local CPU model** (no OpenAI / cloud vision)
3. Store the prediction + a copy of the tile
4. Let an analyst query stored results (`GET /tiles`, `GET /tiles/{id}`)

Everything else from a full product is stubbed. That is intentional.
The brief asked for a design + a thin backend slice, not a finished
product and not a hosted “AI modal”.

## Tech used

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| API | FastAPI + Uvicorn |
| Model | scikit-learn RandomForest, trained on the provided `candidate_tiles` |
| Features | colour histograms + 8×8 spatial thumbnail + channel stats (Pillow / NumPy) |
| Storage | SQLite for metadata, files on disk for the tiles |
| Runtime | fully offline after `pip install` and `python scripts/train.py` |

Why this stack: the brief says use what you know, run offline, and they
are not grading accuracy. This is small enough to explain and change live,
and the classifier is behind one interface so a torch/ONNX model can replace
it later without touching the API or the database.

## Setup

```powershell
cd C:\Users\anura\galaxyeye
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train.py
```

`scripts/train.py` fits the model on `data/be-mlsys-assignment-dataset/candidate_tiles`
and prints accuracy on `eval_set` (labels in `eval_labels.csv`). The artifact
is written to `artifacts/landuse_clf.joblib`.

On this zip the model is about **77% on eval_set**. Highway is the weak class
(often confused with Residential / River). That is expected and useful for
Part 3 — they are not grading accuracy.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Core path — classify one tile and store it:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/tiles" -F "file=@data/be-mlsys-assignment-dataset/eval_set/tile_001.png"
```

```text
POST /tiles                      one tile
POST /tiles/batch                many tiles
POST /ingest/eval-set            classify the assignment folder
GET  /tiles?label=&review_status=
GET  /tiles/{id}
GET  /tiles/{id}/image
PATCH /tiles/{id}/review         human corrected_label
POST /canary   GET /canary
GET  /health
```

Canary: `POST /canary` or the Model page (writes `var/canary.json`).

Interactive docs: http://127.0.0.1:8000/docs

## Dataset

Assignment zip, EuroSAT subset (Helber et al., MIT; modified Copernicus Sentinel-2):

- 7 classes: Forest, River, Residential, Industrial, AnnualCrop, SeaLake, Highway
- `candidate_tiles/`: 150 labelled PNGs per class (used for training)
- `eval_set/`: 210 unlabelled-at-inference tiles; truth is in `eval_labels.csv`

## Still not built (on purpose)

Postgres, login, a map, a GPU CNN, multi-process workers. Those fight the
offline-appliance brief.

---

## Part 3 — problem-solving

### 1. The classifier is wrong ~30% of the time. What do I do? Is it useful?

I would not start by retraining in a panic. First I would ask **which
errors, for which class, and what they cost**.

On this dataset I would use the eval set they already gave us: confusion
matrix, per-class precision/recall, and a reliability view (accuracy in
confidence buckets). “30% wrong” is an average. If almost all mistakes
are Highway ↔ Industrial and Forest is clean, the service may already be
useful for a forest-cover question and useless for infrastructure.

“Good enough” is a product test, not a model test:

- What decision does the analyst make with a label?
- What is the cost of a false Industrial vs a missed River?
- After we hold out low-confidence tiles for review, what is the error
  rate on the **auto-accepted** slice?

If auto-accepted error is low and the review pile is a size a human can
clear, 70% raw accuracy can still be useful. If every class is noisy and
confidence is uncalibrated, it is not useful as automation — only as a
sort hint.

What I would actually do: keep storing full scores, raise the threshold
so the machine only auto-accepts when it is usually right, send the rest
to review, and collect those corrections as the next training set. A new
backbone comes after I know which confusion I am paying for.

### 2. It runs offline, nobody is watching. A month later, how do I know it still works?

I would not rely on a dashboard in the cloud. I would make the box
**emit evidence on disk**.

Minimum I would ship:

1. A heartbeat file / health record: process up, model file hash, last
   successful inference time, disk free, SQLite integrity check.
2. A **canary set**: 20–30 gold tiles (or the eval set) scored on a
   schedule (cron / Windows task). Write accuracy and the confusion
   matrix next to a baseline captured at deploy time. If accuracy drops
   or one class collapses, the box is sick even if `/tiles` still 200s.
3. Structured local logs: request id, model version, latency, predicted
   label, confidence. Rotate them. A month of silence with no canary is
   not “healthy”.
4. A watchdog that restarts a dead process and leaves a crash breadcrumb.

`GET /health` in this slice is the start of that: model loaded, version,
eval accuracy at train time, last inference, counts by label and review
status. In production I would have something read that locally and fail
a scheduled job if the canary moves.

The failure mode I care about offline is **silent wrong**, not crash.
Crashes are easy. A model file that was overwritten, a feature bug, or
new-looking tiles is what a canary is for.

### 3. Tiles arrive, stored results look wrong. Steps in order.

I would split the world into **wrong image, wrong model, wrong write**.

1. Pick one bad id from SQLite. Read `filename`, `predicted_label`,
   `scores`, `model_version`, `tile_path`, `created_at`.
2. Open the stored file at `tile_path`. Is it the tile the analyst
   thinks it is? If not, ingest / filename mix-up. Stop. Do not touch
   the model yet.
3. Re-run `LandUseClassifier` on those exact bytes. Same label and
   scores? If yes, the live path is consistent. If no, we have two
   artifacts or a preprocess drift (resize, RGB vs RGBA).
4. Check `model_version` against the joblib we think is deployed. Hash
   the file. If someone retrained and copied a new artifact, that is
   the incident.
5. Look at the full score vector. Confident and wrong vs 0.34 / 0.33 /
   0.31 is a different bug. The second is thresholding / review, not
   “the database is corrupt”.
6. Run the eval set through the same binary. If eval is still at the
   train-time number, the model is fine and the new tiles are
   out-of-distribution (or the analyst’s expected label is wrong). If
   eval has also collapsed, the code or artifact changed.
7. Only then look at training: class mapping, folder names, label file.
8. Last: storage. `PRAGMA integrity_check`, “did we update the wrong
   row”, “are we reading a different DB file than we write”.

I would not start by opening `train.py` and tweaking hyperparameters.
Most “the results look wrong” bugs in this design are path, version, or
expectation bugs.

### 4. Weakest part of the design, and what breaks it first

The weakest part is the **model, not the API**. Colour histograms and an
8×8 thumbnail will not survive a sensor change, a new GSD, heavy cloud,
night/off-nadir looks, or a class that is defined by texture at a scale
we threw away (Highway vs Industrial is the obvious failure).

What breaks first in the real world: **distribution shift**. The day
someone feeds a different satellite, 256×256 chips, or a new geography,
confidence stays high and labels go bad. The review flag will not save
us if the model is confidently wrong.

Second weak point: one global confidence threshold and no human-review
write-back in this slice. We can mark `needs_review`; we cannot yet
close the loop.

Third: SQLite + a single process. Fine for the assignment. The first
ops break is concurrent ingest or a full disk (we write every PNG
again). I would notice that in the health/canary story before I
rewrote the model.
