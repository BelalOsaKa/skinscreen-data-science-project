"""SkinScreen - AID427 Phase 1 pipeline (PAD-UFES-20).

Usage
-----
    python run_pipeline.py                      # full run
    python run_pipeline.py --fast               # quick run (smaller searches)
    python run_pipeline.py --data-dir /path/to/pad-ufes-20
    python run_pipeline.py --cnn                # + ResNet-50 embeddings (needs torch)
    python run_pipeline.py --smote              # SMOTE inside the training folds
    python run_pipeline.py --skip eda reduction # skip some phases

Phases: eda, reduction, clustering, selection, models (always), report (always)
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_loading import load_dataset, missing_indicator_cols
from src.image_features import attach_image_features
from src.preprocessing import build_preprocessor, patient_split, split_summary
from src.utils import log, save_table, timer

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
from sklearn.exceptions import ConvergenceWarning  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR)
    p.add_argument("--fast", action="store_true", help="smaller searches for a quick check")
    p.add_argument("--cnn", action="store_true", help="add ResNet-50 image embeddings")
    p.add_argument("--smote", action="store_true", help="SMOTE oversampling inside training folds")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["eda", "reduction", "clustering", "selection"])
    return p.parse_args()


def main():
    args = parse_args()
    config.set_output_dir(args.output_dir)
    config.ensure_dirs()
    results: dict = {}

    # ------------------------------------------------------------------ data
    with timer("Data loading + image features"):
        df, miss_report = load_dataset(args.data_dir)
        df, image_cols = attach_image_features(df, use_cnn=args.cnn)
    ind_cols = missing_indicator_cols(df)
    classes = config.CLASSES
    df = df[df[config.TARGET].isin(classes)].reset_index(drop=True)

    results["overview"] = pd.Series({
        "images": len(df), "patients": df[config.GROUP].nunique(),
        "lesions": df["lesion_id"].nunique(),
        "clinical features (raw)": len(config.TABULAR_COLS),
        "missing-value indicators": len(ind_cols),
        "image features": len(image_cols)}).to_frame("value")
    results["class_counts"] = df[config.TARGET].value_counts().reindex(classes).to_frame("count")

    # -------------------------------------------------------------------- EDA
    if "eda" not in args.skip:
        from src.eda import run_eda
        with timer("EDA"):
            run_eda(df, miss_report)

    # ------------------------------------------------------------------ split
    train, test = patient_split(df)
    results["split"] = split_summary(train, test)
    save_table(results["split"], "split_summary")
    ytr = train[config.TARGET].map(classes.index).to_numpy()
    yte = test[config.TARGET].map(classes.index).to_numpy()
    ytr_lbl = train[config.TARGET].to_numpy()

    def make_prep(tab=True, img=True):
        return build_preprocessor(df, image_cols, ind_cols, include_tabular=tab, include_image=img)

    prep = make_prep()
    Xtr = prep.fit_transform(train)
    Xte = prep.transform(test)
    log(f"Pre-processed feature matrix: train {Xtr.shape}, test {Xte.shape}")
    save_table(pd.Series(Xtr.columns, name="feature"), "preprocessed_feature_names", index=False)

    # -------------------------------------------------------------- reduction
    embeddings, emb_idx = {}, {}
    if "reduction" not in args.skip:
        from src import reduction
        with timer("Feature reduction"):
            lin = reduction.linear_reduction(Xtr, ytr_lbl)
            nonlin, emb_idx, quality = reduction.nonlinear_reduction(Xtr, ytr_lbl, fast=args.fast)
            results["linear_summary"] = pd.read_csv(config.TABLE_DIR / "red_linear_summary.csv",
                                                    index_col=0)
            results["nonlinear_quality"] = quality
            results["downstream"] = reduction.downstream_comparison(Xtr, ytr, Xte, yte, fast=args.fast)
            embeddings = {**lin, **nonlin}

    # ------------------------------------------------------------- clustering
    if "clustering" not in args.skip and embeddings:
        from src.clustering import run_clustering
        with timer("Clustering"):
            results["clustering"] = run_clustering(Xtr, ytr_lbl, embeddings, emb_idx)

    # -------------------------------------------------------------- selection
    selected_cols, best_subset = list(Xtr.columns), "All features"
    if "selection" not in args.skip:
        from src.feature_selection import run_feature_selection
        with timer("Feature selection"):
            subsets, fs_comp, best_subset = run_feature_selection(
                Xtr, ytr, train[config.GROUP].to_numpy(), Xte, yte, fast=args.fast)
            selected_cols = subsets[best_subset]
            results["fs_comparison"] = fs_comp
    results["best_subset"] = best_subset

    # ----------------------------------------------------------------- models
    from src.evaluation import evaluate_models, modality_comparison
    from src.models import get_models
    with timer("Models"):
        models = get_models(fast=args.fast)
        ev = evaluate_models(models, prep, selected_cols, train, ytr, test, yte, smote=args.smote)
        results["evaluation"] = ev
        results["modality"] = modality_comparison(models[ev["best"]], make_prep, train, ytr,
                                                  test, yte, selected_cols, ev["best"])

    # ----------------------------------------------------------------- report
    from src.report import related_work_table, write_report
    b = ev["holdout"].loc[ev["best"]]
    results["related"] = related_work_table({
        "model": ev["best"], "selection": best_subset,
        "accuracy": b["test accuracy"], "bacc": b["test balanced accuracy"],
        "f1": b["test F1 (macro)"], "auc": b["test ROC-AUC (OvR macro)"]})
    for key in ("linear_summary", "nonlinear_quality", "downstream", "clustering", "fs_comparison"):
        results.setdefault(key, pd.DataFrame({"note": ["phase skipped"]}))
    if isinstance(results["clustering"].index, pd.MultiIndex) is False:
        results["clustering"] = pd.DataFrame(
            {"note": ["phase skipped"]},
            index=pd.MultiIndex.from_tuples([("-", "KMeans")], names=["space", "algorithm"]))
    write_report(results)
    log("All done.")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
