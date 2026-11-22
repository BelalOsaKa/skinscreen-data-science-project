"""Clustering before and after dimensionality reduction.

"Before": cluster in the full pre-processed feature space.
"After" : cluster in each 2-D embedding produced by reduction.py.
Internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz) measure how
compact/separated the clusters are; external metrics (ARI, NMI, purity)
measure how well the clusters match the 6 diagnoses.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture

from . import config
from .reduction import _scatter
from .utils import log, save_fig, save_table


def purity(y_true, y_pred) -> float:
    cm = contingency_matrix(y_true, y_pred)
    return cm.max(axis=0).sum() / cm.sum()


def _algorithms(k: int):
    rs = config.RANDOM_STATE
    return {
        "KMeans": lambda: KMeans(k, n_init=10, random_state=rs),
        "Agglomerative (Ward)": lambda: AgglomerativeClustering(k, linkage="ward"),
        "GMM": lambda: GaussianMixture(k, covariance_type="full", random_state=rs),
    }


def _scores(X, y, labels) -> dict:
    ok = len(np.unique(labels)) > 1
    return {
        "silhouette": silhouette_score(X, labels) if ok else np.nan,
        "davies_bouldin": davies_bouldin_score(X, labels) if ok else np.nan,
        "calinski_harabasz": calinski_harabasz_score(X, labels) if ok else np.nan,
        "ARI": adjusted_rand_score(y, labels),
        "NMI": normalized_mutual_info_score(y, labels),
        "purity": purity(y, labels),
    }


def choose_k(X: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """Elbow (inertia) and silhouette curves for KMeans in the original space."""
    rows = []
    for k in k_range:
        km = KMeans(k, n_init=10, random_state=config.RANDOM_STATE).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    res = pd.DataFrame(rows).set_index("k")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(res.index, res["inertia"], marker="o")
    ax[0].set_title("Elbow method (KMeans inertia)")
    ax[1].plot(res.index, res["silhouette"], marker="o", color="#DD8452")
    ax[1].set_title("Silhouette vs k")
    for a in ax:
        a.axvline(len(config.CLASSES), ls="--", color="gray", lw=0.8)
        a.set_xlabel("k")
    save_fig(fig, "clu_01_choose_k")
    save_table(res.round(4), "clu_choose_k")
    return res


def run_clustering(X_full: pd.DataFrame, y: np.ndarray, embeddings: dict[str, np.ndarray],
                   emb_index: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    log("=== Clustering before / after reduction ===")
    k = len(config.CLASSES)
    choose_k(X_full.values)

    spaces = {"Original (no reduction)": (X_full.values, np.arange(len(y)))}
    for name, emb in embeddings.items():
        idx = emb_index.get(name, np.arange(len(y))) if emb_index else np.arange(len(y))
        spaces[f"{name} 2-D"] = (emb, idx)

    rows, km_labels = [], {}
    for space, (Z, idx) in spaces.items():
        for alg, make in _algorithms(k).items():
            try:
                lab = make().fit_predict(Z)
            except Exception as e:
                log(f"  {alg} on {space} failed: {e}")
                continue
            rows.append({"space": space, "algorithm": alg, **_scores(Z, y[idx], lab)})
            if alg == "KMeans":
                km_labels[space] = lab
    res = pd.DataFrame(rows).set_index(["space", "algorithm"]).round(4)
    save_table(res, "clu_metrics_before_after")
    log("  best NMI (KMeans): " + ", ".join(
        f"{s}={v:.3f}" for s, v in res.xs("KMeans", level="algorithm")["NMI"]
        .sort_values(ascending=False).head(3).items()))

    # ---- figure: true labels vs KMeans clusters for each 2-D space --------- #
    emb_spaces = [s for s in km_labels if s != "Original (no reduction)"]
    if emb_spaces:
        fig, axes = plt.subplots(len(emb_spaces), 2, figsize=(10, 4.2 * len(emb_spaces)))
        axes = np.atleast_2d(axes)
        for r, space in enumerate(emb_spaces):
            Z, idx = spaces[space]
            _scatter(axes[r, 0], Z, y[idx], f"{space}: true diagnosis")
            lab = km_labels[space]
            pal = dict(zip(sorted(np.unique(lab)), sns.color_palette("tab10", len(np.unique(lab)))))
            for c in sorted(np.unique(lab)):
                m = lab == c
                axes[r, 1].scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7, color=pal[c], label=f"C{c}")
            axes[r, 1].set_title(f"{space}: KMeans (k={k})", fontsize=10)
            axes[r, 1].set_xticks([])
            axes[r, 1].set_yticks([])
        axes[0, 0].legend(markerscale=2, fontsize=8)
        fig.suptitle("Visualisation after reduction and clustering")
        save_fig(fig, "clu_02_embeddings_vs_clusters")

    # ---- figure: metric comparison --------------------------------------- #
    km = res.xs("KMeans", level="algorithm")
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * len(km) + 2))
    km[["ARI", "NMI", "purity"]].plot(kind="barh", ax=axes[0])
    axes[0].set_title("KMeans vs true diagnosis (external)")
    km[["silhouette"]].plot(kind="barh", ax=axes[1], color="#55A868", legend=False)
    axes[1].set_title("KMeans silhouette (internal)")
    for a in axes:
        a.invert_yaxis()
    axes[1].set_yticklabels([])
    save_fig(fig, "clu_03_metrics_before_after")

    # ---- contingency of best space --------------------------------------- #
    best = km["NMI"].idxmax()
    Z, idx = spaces[best]
    ct = pd.crosstab(pd.Series(y[idx], name="diagnosis"),
                     pd.Series(km_labels[best], name="cluster"))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Clusters vs diagnoses ({best}, KMeans)")
    save_fig(fig, "clu_04_contingency_best")
    return res
