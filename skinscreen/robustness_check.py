"""Robustness check: does the model rely on the 'missing metadata' shortcut?

In PAD-UFES-20, every image whose clinical questionnaire is missing belongs to
a NON-cancer class (ACK, NEV or SEK); all BCC, SCC and MEL images have complete
metadata. The missingness pattern itself therefore predicts "non-cancer". It
reaches the model through the `<col>_missing` indicators, the categorical
'MISSING' level and the identical imputed values.

This script measures how much of the Phase 1 performance comes from that
shortcut. It compares, for clinical-only, image-only and fused features:

  A. the full dataset, as in run_pipeline.py;
  B. only the images whose metadata is complete, where the shortcut cannot
     exist (a fair estimate of how well each modality really works).

Run it after run_pipeline.py (it reuses the cached image features, so the
images themselves are optional):

    python robustness_check.py
    python robustness_check.py --data-dir /path/to/pad-ufes-20
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_validate

from src import config
from src.data_loading import load_dataset, missing_indicator_cols
from src.evaluation import compute_metrics
from src.image_features import attach_image_features
from src.models import build_pipeline, get_models
from src.preprocessing import build_preprocessor, group_cv_splits, patient_split
from src.utils import CLASS_COLORS, log, save_fig, save_table

warnings.filterwarnings("ignore")

# The 11 clinical fields that are blank together for the same 804 images
# (the patient questionnaire was not recorded for them).
QUESTIONNAIRE = ["smoke", "drink", "pesticide", "gender", "skin_cancer_history",
                 "cancer_history", "has_piped_water", "has_sewage_system",
                 "fitspatrick", "diameter_1", "diameter_2"]

MODELS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "Hist Gradient Boosting"]
MODALITIES = {"Clinical only": (True, False), "Image only": (False, True),
              "Fusion (clinical + image)": (True, True)}


def missing_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Share of images per class whose clinical questionnaire is missing."""
    miss = df[QUESTIONNAIRE].isna().all(axis=1).map({False: "metadata present",
                                                       True: "metadata missing"})
    tbl = (pd.crosstab(df[config.TARGET], miss)
           .reindex(index=config.CLASSES, columns=["metadata present", "metadata missing"])
           .fillna(0).astype(int))
    tbl.columns.name, tbl.index.name = None, "diagnosis"
    tbl["% missing"] = (100 * tbl["metadata missing"] / tbl.sum(axis=1)).round(1)
    tbl["cancer"] = [c in config.CANCER_CLASSES for c in tbl.index]
    return tbl


def plot_missing_by_class(tbl: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(tbl))
    ax.bar(x, tbl["metadata present"], color="#4C72B0", label="metadata present")
    ax.bar(x, tbl["metadata missing"], bottom=tbl["metadata present"], color="#C44E52",
           label="metadata missing")
    for i, (cls, row) in enumerate(tbl.iterrows()):
        ax.text(i, row["metadata present"] + row["metadata missing"] + 10,
                f"{row['% missing']:.0f}% missing", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{c}\n({'cancer' if tbl.loc[c, 'cancer'] else 'non-cancer'})"
                      for c in tbl.index])
    ax.set_ylim(0, 1.15 * (tbl["metadata present"] + tbl["metadata missing"]).max())
    ax.set_ylabel("images")
    ax.set_title("Missing clinical metadata occurs only in non-cancer classes")
    ax.legend()
    save_fig(fig, "rob_01_missing_by_class")


def evaluate(df, image_cols, ind_cols, train, test, label):
    C = config.CLASSES
    ytr = train[config.TARGET].map(C.index).to_numpy()
    yte = test[config.TARGET].map(C.index).to_numpy()
    cv = group_cv_splits(ytr, train[config.GROUP])
    models = get_models(fast=True)
    rows = []
    for mod, (tab, img) in MODALITIES.items():
        prep = build_preprocessor(df, image_cols, ind_cols, include_tabular=tab, include_image=img)
        cols = list(prep.fit(train).get_feature_names_out())
        for name in MODELS:
            pipe = build_pipeline(prep, cols, clone(models[name]))
            cvr = cross_validate(pipe, train, ytr, cv=cv, scoring="balanced_accuracy",
                                 n_jobs=config.N_JOBS)
            pipe.fit(train, ytr)
            proba = pipe.predict_proba(test)
            m = compute_metrics(yte, proba.argmax(1), proba)
            rows.append({"data": label, "features": mod, "model": name,
                         "train images": len(train), "test images": len(test),
                         "CV balanced acc": cvr["test_score"].mean(),
                         "test accuracy": m["accuracy"],
                         "test balanced acc": m["balanced accuracy"],
                         "test F1 (macro)": m["F1 (macro)"],
                         "test ROC-AUC (OvR macro)": m["ROC-AUC (OvR macro)"],
                         "test cancer sensitivity": m["cancer sensitivity"],
                         "test cancer specificity": m["cancer specificity"]})
            log(f"  [{label}] {mod:<26} {name:<24} test bacc={m['balanced accuracy']:.3f}")
    return rows


def plot_comparison(res: pd.DataFrame, model: str) -> None:
    sub = res[res["model"] == model].pivot(index="features", columns="data",
                                           values="test balanced acc")
    sub = sub.reindex(list(MODALITIES))
    fig, ax = plt.subplots(figsize=(8, 4))
    sub.plot(kind="bar", ax=ax, color=["#8C8C8C", "#4C72B0"], rot=0)
    for cont in ax.containers:
        ax.bar_label(cont, fmt="%.2f", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test balanced accuracy")
    ax.set_title(f"{model}: full data vs complete-metadata subset")
    ax.legend(fontsize=8)
    save_fig(fig, "rob_02_full_vs_complete_metadata")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    args = p.parse_args()
    config.set_output_dir(args.output_dir)
    config.ensure_dirs()

    df, _ = load_dataset(args.data_dir, require_images=False)
    df, image_cols = attach_image_features(df)
    ind_cols = missing_indicator_cols(df)

    tbl = missing_by_class(df)
    save_table(tbl, "rob_missing_by_class")
    plot_missing_by_class(tbl)
    print(tbl.to_string())

    train, test = patient_split(df)                       # same split as Phase 1
    complete = lambda d: d[d[QUESTIONNAIRE].notna().all(axis=1)].reset_index(drop=True)
    rows = evaluate(df, image_cols, ind_cols, train, test, "A. Full dataset")
    rows += evaluate(df, image_cols, [], complete(train), complete(test),
                     "B. Complete-metadata subset")
    res = pd.DataFrame(rows).round(4)
    save_table(res, "rob_full_vs_complete_metadata", index=False)
    plot_comparison(res, "Logistic Regression")
    show = ["data", "features", "model", "test balanced acc", "test F1 (macro)",
            "test ROC-AUC (OvR macro)", "test cancer sensitivity"]
    print(res[show].to_string(index=False))
    log("Robustness check done.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
