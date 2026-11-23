"""Exploratory data analysis: class balance, missing values, distributions,
correlations, outliers and image quality ("noise") statistics."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from . import config
from .utils import CLASS_COLORS, log, save_fig, save_table


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().reindex(config.CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bars = axes[0].bar(counts.index, counts.values,
                       color=[CLASS_COLORS[c] for c in counts.index])
    axes[0].bar_label(bars)
    axes[0].set_title("Diagnosis distribution (6 classes)")
    axes[0].set_ylabel("images")
    ratio = counts.max() / counts.min()
    axes[0].text(0.98, 0.95, f"imbalance ratio {ratio:.1f}:1", transform=axes[0].transAxes,
                 ha="right", va="top")

    cancer = df["is_cancer"].map({1: "Cancer (BCC/SCC/MEL)", 0: "Non-cancer (ACK/NEV/SEK)"})
    cc = cancer.value_counts()
    axes[1].pie(cc.values, labels=cc.index, autopct="%1.1f%%", colors=["#C44E52", "#4C72B0"])
    axes[1].set_title("Cancer vs non-cancer")
    axes[1].grid(False)
    save_fig(fig, "eda_01_class_distribution")
    save_table(counts.to_frame("count").assign(pct=lambda d: (100 * d["count"] / d["count"].sum()).round(2)),
               "eda_class_distribution")


def plot_missing(report: pd.DataFrame, n_rows: int) -> None:
    rep = report[report["total_missing"] > 0]
    if rep.empty:
        log("  no missing values")
        return
    blank_pct, unk_pct = 100 * rep["blank"] / n_rows, 100 * rep["UNK"] / n_rows
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(rep))))
    ax.barh(rep.index, blank_pct, color="#8C8C8C", label="blank")
    ax.barh(rep.index, unk_pct, left=blank_pct, color="#DD8452", label="'UNK'")
    ax.invert_yaxis()
    ax.set_xlabel("% of rows")
    ax.set_title("Missing values per clinical feature (blank + 'UNK')")
    ax.legend()
    save_fig(fig, "eda_02_missing_values")
    save_table(report, "eda_missing_values")


def plot_numeric(df: pd.DataFrame) -> pd.DataFrame:
    fig, axes = plt.subplots(2, len(config.NUMERIC_COLS), figsize=(4 * len(config.NUMERIC_COLS), 7))
    outliers = {}
    for j, c in enumerate(config.NUMERIC_COLS):
        sns.histplot(df[c].dropna(), kde=True, ax=axes[0, j], color="#4C72B0")
        axes[0, j].set_title(f"{c} (n={df[c].notna().sum()})")
        sns.boxplot(data=df, x=config.TARGET, y=c, order=config.CLASSES, ax=axes[1, j],
                    hue=config.TARGET, palette=CLASS_COLORS, legend=False)
        axes[1, j].set_xlabel("")
        x = df[c].dropna()
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers[c] = {"n_outliers_1.5IQR": int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum()),
                       "n_outliers_3IQR": int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum()),
                       "min": x.min(), "median": x.median(), "max": x.max(),
                       "mean": x.mean(), "std": x.std(), "skew": x.skew()}
    fig.suptitle("Numerical features: distribution (top) and per diagnosis (bottom)")
    save_fig(fig, "eda_03_numeric_features")
    out = pd.DataFrame(outliers).T.round(3)
    save_table(out, "eda_numeric_summary_outliers")
    return out


def plot_categorical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, c in zip(axes.ravel(), config.CATEGORICAL_COLS):
        ct = pd.crosstab(df[c].fillna("MISSING"), df[config.TARGET]).reindex(columns=config.CLASSES)
        ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index[:12]]
        ct.plot(kind="barh", stacked=True, ax=ax, color=[CLASS_COLORS[k] for k in config.CLASSES],
                legend=ax is axes[0, 0])
        ax.invert_yaxis()
        ax.set_title(c)
        ax.set_ylabel("")
    fig.suptitle("Categorical features by diagnosis (top-12 levels)")
    save_fig(fig, "eda_04_categorical_features")


def plot_binary(df: pd.DataFrame) -> None:
    prev = df.groupby(config.TARGET)[config.BINARY_COLS].mean().reindex(config.CLASSES) * 100
    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(prev, annot=True, fmt=".0f", cmap="Blues", ax=ax,
                cbar_kws={"label": "% 'True' (of answered)"})
    ax.set_title("Binary features: prevalence per diagnosis")
    save_fig(fig, "eda_05_binary_features")
    save_table(prev.round(2), "eda_binary_prevalence")


def plot_correlation(df: pd.DataFrame) -> None:
    cols = config.NUMERIC_COLS + config.BINARY_COLS + ["is_cancer"]
    corr = df[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True,
                fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set_title("Spearman correlation of clinical features")
    save_fig(fig, "eda_06_correlation")


def plot_patients(df: pd.DataFrame) -> None:
    per_patient = df.groupby(config.GROUP).size()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    per_patient.value_counts().sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
    ax.set_xlabel("images per patient")
    ax.set_ylabel("patients")
    ax.set_title(f"{(per_patient > 1).sum()} patients have >1 image -> patient-level split")
    save_fig(fig, "eda_07_images_per_patient")


def plot_image_quality(df: pd.DataFrame) -> None:
    q = [c for c in ("q_width", "q_height", "q_brightness", "q_sharpness") if c in df]
    if not q:
        return
    fig, axes = plt.subplots(1, len(q), figsize=(4.2 * len(q), 3.8))
    for ax, c in zip(np.atleast_1d(axes), q):
        sns.boxplot(data=df, x=config.TARGET, y=c, order=config.CLASSES, ax=ax,
                    hue=config.TARGET, palette=CLASS_COLORS, legend=False)
        ax.set_xlabel("")
        ax.set_title(c.replace("q_", ""))
    fig.suptitle("Image quality / acquisition noise (resolution, brightness, sharpness)")
    save_fig(fig, "eda_08_image_quality")
    save_table(df.groupby(config.TARGET)[q].describe().round(3), "eda_image_quality")


def plot_sample_images(df: pd.DataFrame, n_per_class: int = 5) -> None:
    from .image_features import load_and_preprocess

    fig, axes = plt.subplots(len(config.CLASSES), n_per_class * 2,
                             figsize=(2 * n_per_class * 2, 2.1 * len(config.CLASSES)))
    rng = np.random.default_rng(config.RANDOM_STATE)
    for i, cls in enumerate(config.CLASSES):
        sub = df[df[config.TARGET] == cls]
        picks = sub.iloc[rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)]
        for j in range(n_per_class):
            for k in range(2):
                ax = axes[i, 2 * j + k]
                ax.axis("off")
                if j >= len(picks):
                    continue
                path = picks.iloc[j]["image_path"]
                if k == 0:
                    with Image.open(path) as im:
                        ax.imshow(im.convert("RGB").resize((128, 128)))
                else:
                    ax.imshow(load_and_preprocess(path))
                if j == 0 and k == 0:
                    ax.set_title(cls, loc="left", fontsize=10, fontweight="bold")
    fig.suptitle("Sample images per class: raw (left) vs colour-normalised + denoised (right)")
    save_fig(fig, "eda_09_sample_images")


def run_eda(df: pd.DataFrame, report: pd.DataFrame) -> None:
    log("=== EDA ===")
    overview = pd.Series({
        "images": len(df), "patients": df[config.GROUP].nunique(),
        "lesions": df["lesion_id"].nunique(),
        "clinical_features": len(config.TABULAR_COLS),
        "numeric": len(config.NUMERIC_COLS), "categorical": len(config.CATEGORICAL_COLS),
        "binary": len(config.BINARY_COLS),
        "rows_with_any_missing": int(df[config.TABULAR_COLS].isna().any(axis=1).sum()),
    })
    save_table(overview.to_frame("value"), "eda_overview")
    plot_class_distribution(df)
    plot_missing(report, len(df))
    plot_numeric(df)
    plot_categorical(df)
    plot_binary(df)
    plot_correlation(df)
    plot_patients(df)
    plot_image_quality(df)
    if df["image_path"].notna().any():
        plot_sample_images(df)
    else:
        log("  no image files found: sample-image figure skipped (cached features used)")
