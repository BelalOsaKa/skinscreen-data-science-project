"""Project-wide configuration: paths, column groups, constants."""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Put the extracted PAD-UFES-20 archive here (see data/README.md):
#   data/pad-ufes-20/metadata.csv
#   data/pad-ufes-20/images/imgs_part_1/*.png  (and imgs_part_2, imgs_part_3)
DATA_DIR = PROJECT_ROOT / "data" / "pad-ufes-20"
METADATA_FILE = "metadata.csv"
IMAGES_SUBDIR = "images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CACHE_DIR = OUTPUT_DIR / "cache"

RANDOM_STATE = 42
N_JOBS = -1

# --------------------------------------------------------------------------- #
# Columns of metadata.csv
# --------------------------------------------------------------------------- #
TARGET = "diagnostic"
GROUP = "patient_id"                      # used for patient-level splitting
ID_COLS = ["patient_id", "lesion_id", "img_id"]
LEAKAGE_COLS = ["biopsed"]                # linked to the diagnosis -> dropped

NUMERIC_COLS = ["age", "fitspatrick", "diameter_1", "diameter_2"]
CATEGORICAL_COLS = ["gender", "region", "background_father", "background_mother"]
BINARY_COLS = [
    "smoke", "drink", "pesticide", "skin_cancer_history", "cancer_history",
    "has_piped_water", "has_sewage_system",
    "itch", "grew", "hurt", "changed", "bleed", "elevation",
]
TABULAR_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS

CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_NAMES = {
    "ACK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NEV": "Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "SEK": "Seborrheic Keratosis",
}
CANCER_CLASSES = {"BCC", "SCC", "MEL"}

# --------------------------------------------------------------------------- #
# Image feature extraction
# --------------------------------------------------------------------------- #
IMG_SIZE = 128                 # images are resized to IMG_SIZE x IMG_SIZE
IMG_FEATURE_PREFIX = "img_"
CNN_FEATURE_PREFIX = "cnn_"

# --------------------------------------------------------------------------- #
# Experiment settings
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.20               # 80 % train / 20 % test (patient-level)
N_FOLDS = 5                    # k-fold cross-validation
MAX_EMBED_SAMPLES = 1500       # subsample size for O(n^2) methods (MDS, SNE)


def set_output_dir(path) -> None:
    """Redirect all figures/tables/cache to another folder."""
    global OUTPUT_DIR, FIG_DIR, TABLE_DIR, CACHE_DIR
    OUTPUT_DIR = Path(path).resolve()
    FIG_DIR, TABLE_DIR, CACHE_DIR = OUTPUT_DIR / "figures", OUTPUT_DIR / "tables", OUTPUT_DIR / "cache"


def ensure_dirs() -> None:
    for d in (OUTPUT_DIR, FIG_DIR, TABLE_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
