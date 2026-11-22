"""Load PAD-UFES-20 metadata, link it to the images and clean basic types."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .utils import log


def _to_binary(s: pd.Series) -> pd.Series:
    """Map 'True'/'False' (str or bool) to 1.0/0.0; anything else -> NaN."""
    mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0,
               "1.0": 1.0, "0.0": 0.0}
    return s.map(lambda v: mapping.get(str(v).strip().lower(), np.nan)
                 if pd.notna(v) else np.nan)


def missing_report(raw: pd.DataFrame) -> pd.DataFrame:
    """Count blank (NaN) and 'UNK' answers per clinical column."""
    rows = []
    for c in config.TABULAR_COLS:
        col = raw[c]
        n_blank = int(col.isna().sum())
        n_unk = int((col.astype(str).str.strip().str.upper() == "UNK").sum())
        rows.append({"feature": c, "blank": n_blank, "UNK": n_unk,
                     "total_missing": n_blank + n_unk,
                     "pct_missing": 100 * (n_blank + n_unk) / len(col)})
    return (pd.DataFrame(rows).set_index("feature")
            .sort_values("pct_missing", ascending=False))


def find_images(images_dir: Path) -> dict[str, Path]:
    """Map image file name -> full path (images are split in several sub-folders)."""
    paths = {}
    for ext in ("*.png", "*.PNG", "*.jpg", "*.jpeg"):
        for p in images_dir.rglob(ext):
            paths[p.name] = p
    return paths


def load_dataset(data_dir: Path | str = config.DATA_DIR, require_images: bool = True):
    """Return (df, missing_report).

    ``df`` has one row per image, cleaned dtypes, an ``image_path`` column,
    a binary ``is_cancer`` column, and explicit ``<col>_missing`` indicators.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / config.METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Download PAD-UFES-20 from "
            "https://data.mendeley.com/datasets/zr7vgbcyr2/1 and extract it into "
            f"{data_dir} (see data/README.md).")

    raw = pd.read_csv(meta_path, dtype=str, keep_default_na=True)
    log(f"Loaded metadata: {raw.shape[0]} rows x {raw.shape[1]} columns")
    report = missing_report(raw)

    df = raw.copy()
    # 'UNK' means the patient did not know / did not answer -> treat as missing
    df = df.replace({"UNK": np.nan, "unk": np.nan, "": np.nan})

    for c in config.NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in config.BINARY_COLS:
        df[c] = _to_binary(df[c])
    for c in config.CATEGORICAL_COLS:
        df[c] = df[c].str.strip().str.upper()
    df[config.TARGET] = df[config.TARGET].str.strip().str.upper()

    # --- link to images ---------------------------------------------------- #
    images = find_images(data_dir / config.IMAGES_SUBDIR)
    if not images:  # some mirrors put the PNGs directly in data_dir
        images = find_images(data_dir)
    df["image_path"] = df["img_id"].map(lambda n: str(images.get(n, "")) or np.nan)
    n_missing_img = int(df["image_path"].isna().sum())
    if n_missing_img:
        log(f"WARNING: {n_missing_img} metadata rows have no image file")
        if require_images:
            df = df[df["image_path"].notna()].reset_index(drop=True)

    # --- drop duplicates & leakage --------------------------------------- #
    before = len(df)
    df = df.drop_duplicates(subset="img_id").reset_index(drop=True)
    if len(df) != before:
        log(f"Removed {before - len(df)} duplicated img_id rows")
    df = df.drop(columns=[c for c in config.LEAKAGE_COLS if c in df.columns])

    df["is_cancer"] = df[config.TARGET].isin(config.CANCER_CLASSES).astype(int)

    # Explicit missing-value indicators. They are created from the column list
    # (not from the target), so they leak nothing; doing it here keeps the
    # number of model features identical across CV folds.
    for c in config.TABULAR_COLS:
        if df[c].isna().any():
            df[f"{c}_missing"] = df[c].isna().astype(int)

    log(f"Final dataset: {len(df)} images, {df[config.GROUP].nunique()} patients, "
        f"{df['lesion_id'].nunique()} lesions")
    return df, report


def missing_indicator_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_missing")]
