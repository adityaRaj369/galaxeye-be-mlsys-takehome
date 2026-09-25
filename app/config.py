from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "be-mlsys-assignment-dataset"
CANDIDATE_DIR = DATA_DIR / "candidate_tiles"
EVAL_DIR = DATA_DIR / "eval_set"
EVAL_LABELS = DATA_DIR / "eval_labels.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "landuse_clf.joblib"
VAR_DIR = ROOT / "var"
TILE_STORE_DIR = VAR_DIR / "tiles"
DB_PATH = VAR_DIR / "predictions.db"
CANARY_PATH = VAR_DIR / "canary.json"
APP_LOG = VAR_DIR / "app.log"

# EuroSAT subset used in the assignment zip.
CLASSES = [
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
]

# Below this, we still store the top class but mark the row for review.
CONFIDENCE_THRESHOLD = 0.50
IMAGE_SIZE = 64
HIST_BINS = 16
SPATIAL_SIZE = 8
MODEL_VERSION = "landuse-rf-v1"
