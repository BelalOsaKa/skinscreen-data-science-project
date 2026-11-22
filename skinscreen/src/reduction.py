"""Linear and non-linear feature reduction.

Linear     : PCA, LDA, (Truncated) SVD
Non-linear : Kernel PCA, SNE (own NumPy implementation), t-SNE, UMAP,
             metric MDS, non-metric MDS, Isomap, LLE

Every method produces a 2-D embedding for visualisation; methods that can
project unseen data (``transform``) are additionally compared as a
pre-processing step for classification (fit on train, apply to test).
"""
from __future__ import annotations

import inspect
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, TSNE, Isomap, LocallyLinearEmbedding, trustworthiness
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .utils import CLASS_COLORS, log, make_logreg, save_fig, save_table

try:
    import umap  # umap-learn
    HAS_UMAP = True
except ImportError:  # pragma: no cover
    HAS_UMAP = False


# --------------------------------------------------------------------------- #
# SNE (Hinton & Roweis, 2002) - not available in scikit-learn
# --------------------------------------------------------------------------- #
def _conditional_P(D2: np.ndarray, perplexity: float, tol: float = 1e-5,
                   max_iter: int = 60) -> np.ndarray:
    """p_{j|i} with a per-point Gaussian bandwidth found by binary search."""
    n = D2.shape[0]
    P = np.zeros((n, n))
    log_u = np.log(perplexity)
    for i in range(n):
        Di = np.delete(D2[i], i)
        Di = Di - Di.min()  # numerical stability (entropy is shift-invariant)
        beta, lo, hi = 1.0, -np.inf, np.inf
        for _ in range(max_iter):
            Pi = np.exp(-Di * beta)
            s = max(Pi.sum(), 1e-12)
            H = np.log(s) + beta * np.sum(Di * Pi) / s
            Pi = Pi / s
            diff = H - log_u
            if abs(diff) < tol:
                break
            if diff > 0:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = beta / 2 if lo == -np.inf else (beta + lo) / 2
        P[i, np.arange(n) != i] = Pi
    return P


class SNE:
    """Original (asymmetric) Stochastic Neighbour Embedding.

    Cost  C = sum_i KL(P_i || Q_i), Gaussian kernels in both spaces.
    Grad  dC/dy_i = 2 sum_j (p_j|i - q_j|i + p_i|j - q_i|j)(y_i - y_j)
    Optimised with momentum, adaptive gains and early exaggeration.
    """

    def __init__(self, n_components=2, perplexity=30.0, n_iter=600, learning_rate=0.5,
                 early_exaggeration=4.0, max_step=0.5, random_state=config.RANDOM_STATE):
        self.max_step = max_step
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state

    @staticmethod
    def _sq_dists(X):
        s = np.sum(X ** 2, axis=1)
        return np.maximum(s[:, None] + s[None, :] - 2 * X @ X.T, 0)

    def fit_transform(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        P = _conditional_P(self._sq_dists(X), self.perplexity)
        Y = rng.normal(0, 1e-4, size=(n, self.n_components))
        update, gains = np.zeros_like(Y), np.ones_like(Y)
        self.kl_history_ = []
        for it in range(self.n_iter):
            exag = self.early_exaggeration if it < 100 else 1.0
            momentum = 0.5 if it < 250 else 0.8
            num = np.exp(-self._sq_dists(Y))
            np.fill_diagonal(num, 0.0)
            Q = num / np.maximum(num.sum(axis=1, keepdims=True), 1e-12)
            Pe = P * exag
            M = (Pe - Q) + (Pe - Q).T
            grad = 2 * (M.sum(axis=1)[:, None] * Y - M @ Y)
            gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
            gains = np.clip(gains, 0.01, 10.0)
            update = momentum * update - self.learning_rate * gains * grad
            # clip each point's step: the Gaussian output kernel has steep
            # gradients and plain SNE easily diverges without this
            norms = np.linalg.norm(update, axis=1, keepdims=True)
            update = update * np.minimum(1.0, self.max_step / np.maximum(norms, 1e-12))
            Y = Y + update
            Y = Y - Y.mean(axis=0)
            if it % 50 == 0 or it == self.n_iter - 1:
                kl = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(Q, 1e-12)))
                self.kl_history_.append((it, kl))
        self.embedding_ = Y
        return Y


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _mds(metric: bool, n_components: int = 2):
    """MDS constructor compatible with old/new scikit-learn argument names."""
    params = inspect.signature(MDS).parameters
    kw = dict(n_components=n_components, random_state=config.RANDOM_STATE,
              max_iter=300, n_init=1)
    if "metric_mds" in params:          # scikit-learn >= 1.8
        kw["metric_mds"] = metric
        kw["metric"] = "euclidean"
    else:
        kw["metric"] = metric
        kw["dissimilarity"] = "euclidean"
    if "normalized_stress" in params:
        kw["normalized_stress"] = "auto"
    return MDS(**kw)


def _subsample(n: int, max_n: int, y=None, seed: int = config.RANDOM_STATE) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    from sklearn.model_selection import train_test_split
    idx, _ = train_test_split(np.arange(n), train_size=max_n, stratify=y, random_state=seed)
    return np.sort(idx)


def _scatter(ax, emb, labels, title, palette=CLASS_COLORS, s=8):
    for cls in sorted(pd.unique(labels)):
        m = labels == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=s, alpha=0.7, label=cls,
                   color=palette.get(cls) if isinstance(palette, dict) else None,
                   edgecolors="none")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def _embedding_quality(X_hd, emb, y, groups=None) -> dict:
    knn = KNeighborsClassifier(n_neighbors=15)
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    return {
        "trustworthiness@10": trustworthiness(X_hd, emb, n_neighbors=10),
        "silhouette(true labels)": silhouette_score(emb, y),
        "kNN balanced acc (5-fold)": cross_val_score(knn, emb, y, cv=cv,
                                                     scoring="balanced_accuracy").mean(),
    }


# --------------------------------------------------------------------------- #
# Linear methods
# --------------------------------------------------------------------------- #
def linear_reduction(X: pd.DataFrame, y: np.ndarray) -> dict[str, np.ndarray]:
    log("=== Linear feature reduction (PCA, LDA, SVD) ===")
    Xv = X.values
    out = {}

    # ---- PCA ---------------------------------------------------------------
    pca = PCA(random_state=config.RANDOM_STATE).fit(Xv)
    cum = np.cumsum(pca.explained_variance_ratio_)
    k90, k95 = int(np.searchsorted(cum, 0.90) + 1), int(np.searchsorted(cum, 0.95) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    kmax = min(60, len(cum))
    ax[0].bar(range(1, kmax + 1), pca.explained_variance_ratio_[:kmax], color="#4C72B0")
    ax[0].set_title("PCA scree plot")
    ax[0].set_xlabel("component")
    ax[0].set_ylabel("explained variance ratio")
    ax[1].plot(range(1, len(cum) + 1), cum, color="#DD8452")
    for k, lvl in ((k90, 0.90), (k95, 0.95)):
        ax[1].axhline(lvl, ls="--", color="gray", lw=0.8)
        ax[1].axvline(k, ls=":", color="gray", lw=0.8)
        ax[1].text(k, lvl - 0.07, f" {k} PCs -> {lvl:.0%}", fontsize=8)
    ax[1].set_title(f"Cumulative explained variance ({X.shape[1]} features)")
    ax[1].set_xlabel("number of components")
    save_fig(fig, "red_01_pca_variance")
    out["PCA"] = pca.transform(Xv)[:, :2]

    # Top loadings of PC1 / PC2
    load = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    top = load.abs().sum(axis=1).sort_values(ascending=False).index[:20]
    fig, ax = plt.subplots(figsize=(7, 6))
    load.loc[top].plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Top-20 feature loadings on PC1 / PC2")
    save_fig(fig, "red_02_pca_loadings")

    # ---- LDA ---------------------------------------------------------------
    lda = LinearDiscriminantAnalysis(solver="svd").fit(Xv, y)
    out["LDA"] = lda.transform(Xv)[:, :2]
    lda_var = pd.Series(lda.explained_variance_ratio_,
                        index=[f"LD{i+1}" for i in range(len(lda.explained_variance_ratio_))])

    # ---- SVD (no centring -> differs from PCA) -----------------------------
    svd = TruncatedSVD(n_components=min(50, X.shape[1] - 1),
                       random_state=config.RANDOM_STATE).fit(Xv)
    out["SVD"] = svd.transform(Xv)[:, :2]

    summary = pd.DataFrame({
        "method": ["PCA", "LDA", "SVD"],
        "type": ["unsupervised, linear", "supervised, linear (max C-1=5 comps)",
                 "unsupervised, linear, no centring"],
        "var. explained by 2 comps": [cum[1], lda_var.iloc[:2].sum(),
                                      svd.explained_variance_ratio_[:2].sum()],
        "comps for 95% var": [k95, "n/a", int(np.searchsorted(
            np.cumsum(svd.explained_variance_ratio_), 0.95) + 1)
            if svd.explained_variance_ratio_.sum() >= 0.95 else ">50"],
    }).set_index("method")
    save_table(summary, "red_linear_summary")
    save_table(lda_var.to_frame("explained_variance_ratio"), "red_lda_variance")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, emb) in zip(axes, out.items()):
        _scatter(ax, emb, y, name)
    axes[0].legend(markerscale=2, fontsize=8)
    fig.suptitle("Linear reduction to 2-D (training set, coloured by diagnosis)")
    save_fig(fig, "red_03_linear_2d")
    return out


# --------------------------------------------------------------------------- #
# Non-linear methods
# --------------------------------------------------------------------------- #
def nonlinear_reduction(X: pd.DataFrame, y: np.ndarray, fast: bool = False):
    """Return (embeddings dict, sample-index dict, quality table)."""
    log("=== Non-linear feature reduction ===")
    Xv = X.values
    n = len(Xv)
    # PCA pre-reduction to 50 dims is standard practice before neighbour-graph
    # methods: it removes noise dimensions and speeds things up.
    X50 = PCA(n_components=min(50, Xv.shape[1]), random_state=config.RANDOM_STATE).fit_transform(Xv)
    sub = _subsample(n, 600 if fast else config.MAX_EMBED_SAMPLES, y)
    sub_sne = _subsample(n, 500 if fast else 1000, y)
    rs = config.RANDOM_STATE

    methods = {
        "Kernel PCA (RBF)": (lambda: KernelPCA(2, kernel="rbf", gamma=1.0 / Xv.shape[1],
                                               random_state=rs), Xv, None),
        "Kernel PCA (poly)": (lambda: KernelPCA(2, kernel="poly", degree=3,
                                                gamma=1.0 / Xv.shape[1], random_state=rs), Xv, None),
        "SNE": (lambda: SNE(perplexity=30, n_iter=300 if fast else 600), X50, sub_sne),
        "t-SNE": (lambda: TSNE(2, perplexity=30, init="pca", random_state=rs,
                               learning_rate="auto"), X50, None),
        "Metric MDS": (lambda: _mds(metric=True), X50, sub),
        "Non-metric MDS": (lambda: _mds(metric=False), X50, sub),
        "Isomap": (lambda: Isomap(n_neighbors=12, n_components=2), X50, None),
        "LLE": (lambda: LocallyLinearEmbedding(n_neighbors=12, n_components=2,
                                               random_state=rs, eigen_solver="dense"), X50, None),
    }
    if HAS_UMAP:
        methods["UMAP"] = (lambda: umap.UMAP(n_neighbors=15, min_dist=0.1,
                                             random_state=rs), X50, None)
    else:
        log("  umap-learn not installed -> UMAP skipped")

    embeds, idxs, fitted, rows = {}, {}, {}, []
    for name, (make, data, idx) in methods.items():
        idx = np.arange(n) if idx is None else idx
        t0 = time.perf_counter()
        try:
            model = make()
            emb = np.asarray(model.fit_transform(data[idx]))
            if not np.isfinite(emb).all():
                raise ValueError("embedding contains NaN/inf")
            fitted[name] = model
        except Exception as e:  # keep the pipeline running
            log(f"  {name} failed: {e}")
            continue
        dt = time.perf_counter() - t0
        embeds[name], idxs[name] = emb, idx
        q = _embedding_quality(X50[idx], emb, y[idx])
        rows.append({"method": name, "n_samples": len(idx), "time_s": round(dt, 1), **q})
        log(f"  {name:<18} n={len(idx):<5} {dt:6.1f}s  kNN-BAcc={q['kNN balanced acc (5-fold)']:.3f}")

    quality = pd.DataFrame(rows).set_index("method").round(4)
    save_table(quality, "red_nonlinear_quality")

    k = len(embeds)
    ncol = 3
    nrow = int(np.ceil(k / ncol))
    for color_by in ("diagnosis", "cancer"):
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.3 * ncol, 4.6 * nrow))
        for ax, (name, emb) in zip(axes.ravel(), embeds.items()):
            lab = y[idxs[name]]
            if color_by == "cancer":
                lab = np.where(np.isin(lab, list(config.CANCER_CLASSES)), "cancer", "non-cancer")
                _scatter(ax, emb, lab, name, palette={"cancer": "#C44E52", "non-cancer": "#4C72B0"})
            else:
                _scatter(ax, emb, lab, name)
        for ax in axes.ravel()[k:]:
            ax.axis("off")
        axes.ravel()[0].legend(markerscale=2, fontsize=8)
        fig.suptitle(f"Non-linear reduction to 2-D (training set, coloured by {color_by})")
        save_fig(fig, f"red_04_nonlinear_2d_{color_by}")

    if "SNE" in fitted and fitted["SNE"].kl_history_:
        it, kl = zip(*fitted["SNE"].kl_history_)
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.plot(it, kl, marker="o", color="#4C72B0")
        ax.axvline(100, ls="--", color="gray", lw=0.8)
        ax.text(102, max(kl), "end of early exaggeration", fontsize=8, va="top")
        ax.set_xlabel("iteration")
        ax.set_ylabel("KL(P || Q)")
        ax.set_title("SNE (own implementation): cost convergence")
        save_fig(fig, "red_06_sne_convergence")
    return embeds, idxs, quality


# --------------------------------------------------------------------------- #
# Reduction as a pre-processing step for classification
# --------------------------------------------------------------------------- #
def downstream_comparison(Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, yte,
                          n_components: int = 10, fast: bool = False) -> pd.DataFrame:
    """Fit each reducer on the training set, project the test set, train a
    classifier on the reduced features and report test scores."""
    log("=== Reduction -> classification (fit on train, evaluate on test) ===")
    rs = config.RANDOM_STATE
    d = Xtr.shape[1]
    reducers = {
        "None (all features)": None,
        "PCA": PCA(n_components, random_state=rs),
        "LDA": LinearDiscriminantAnalysis(n_components=min(5, len(np.unique(ytr)) - 1)),
        "SVD": TruncatedSVD(n_components, random_state=rs),
        "Kernel PCA (RBF)": KernelPCA(n_components, kernel="rbf", gamma=1.0 / d, random_state=rs),
        "Isomap": make_pipeline(PCA(50, random_state=rs), Isomap(n_neighbors=12, n_components=n_components)),
        "LLE": make_pipeline(PCA(50, random_state=rs), LocallyLinearEmbedding(
            n_neighbors=max(12, n_components + 2), n_components=n_components,
            random_state=rs, eigen_solver="dense")),
    }
    if HAS_UMAP and not fast:
        reducers["UMAP"] = make_pipeline(PCA(50, random_state=rs),
                                         umap.UMAP(n_components=n_components, random_state=rs))
    rows = []
    for name, red in reducers.items():
        try:
            if red is None:
                Ztr, Zte = Xtr.values, Xte.values
            else:
                Ztr = red.fit_transform(Xtr.values, ytr) if name == "LDA" else red.fit_transform(Xtr.values)
                Zte = red.transform(Xte.values)
        except Exception as e:
            log(f"  {name} failed: {e}")
            continue
        clf = make_pipeline(StandardScaler(), make_logreg("l2", C=1.0))
        clf.fit(Ztr, ytr)
        pred = clf.predict(Zte)
        rows.append({"reduction": name, "dims": Ztr.shape[1],
                     "test accuracy": accuracy_score(yte, pred),
                     "test balanced acc": balanced_accuracy_score(yte, pred),
                     "test macro F1": f1_score(yte, pred, average="macro"),
                     "train accuracy": accuracy_score(ytr, clf.predict(Ztr))})
    res = pd.DataFrame(rows).set_index("reduction").round(4)
    save_table(res, "red_downstream_comparison")

    fig, ax = plt.subplots(figsize=(9, 4))
    res[["test accuracy", "test balanced acc", "test macro F1"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Logistic regression on reduced features ({n_components} dims; LDA {min(5, len(np.unique(ytr)) - 1)})")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "red_05_downstream_comparison")
    return res
