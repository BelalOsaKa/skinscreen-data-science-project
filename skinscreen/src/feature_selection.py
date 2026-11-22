"""Feature selection.

* Filter     : variance threshold, correlation redundancy, ANOVA F-test,
               mutual information, chi-square -> aggregated rank
* Wrapper    : sequential forward selection, sequential backward selection,
               recursive feature elimination (RFE + RFECV)
* Embedded   : L1 (Lasso), L2 (Ridge), Elastic-Net logistic regression,
               random-forest impurity importance
* Metaheuristic: Genetic Algorithm, Binary Particle Swarm Optimisation,
               Simulated Annealing (all own implementations)

All selection is done on the TRAINING set only, with patient-grouped inner
cross-validation. The test set is only used afterwards to report how each
selected subset generalises.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, RFECV, SequentialFeatureSelector,
                                       VarianceThreshold, chi2, f_classif,
                                       mutual_info_classif)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from . import config
from .preprocessing import group_cv_splits
from .utils import log, make_logreg, save_fig, save_table

SCORING = "balanced_accuracy"   # robust to the 16:1 class imbalance


def _base_estimator():
    return make_logreg("l2", C=1.0, max_iter=500)


# --------------------------------------------------------------------------- #
# Filter methods
# --------------------------------------------------------------------------- #
def filter_methods(X: pd.DataFrame, y, k: int = 30):
    log("--- Filter methods ---")
    cols = X.columns

    vt = VarianceThreshold(threshold=0.01).fit(X)
    low_var = list(cols[~vt.get_support()])

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    redundant = [c for c in upper.columns if (upper[c] > 0.95).any()]

    f_scores, _ = f_classif(X, y)
    mi = mutual_info_classif(X, y, random_state=config.RANDOM_STATE)
    chi, _ = chi2(MinMaxScaler().fit_transform(X), y)

    scores = pd.DataFrame({"ANOVA_F": f_scores, "mutual_info": mi, "chi2": chi},
                          index=cols).fillna(0)
    ranks = scores.rank(ascending=False)
    scores["mean_rank"] = ranks.mean(axis=1)
    scores["low_variance"] = scores.index.isin(low_var)
    scores["redundant(|r|>0.95)"] = scores.index.isin(redundant)
    scores = scores.sort_values("mean_rank")
    save_table(scores.round(5), "fs_filter_scores")

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.5))
    for ax, m, color in zip(axes, ["ANOVA_F", "mutual_info", "chi2"],
                            ["#4C72B0", "#DD8452", "#55A868"]):
        top = scores[m].sort_values(ascending=False).head(20)
        ax.barh(top.index, top.values, color=color)
        ax.invert_yaxis()
        ax.set_title(f"Top-20 by {m}")
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Filter feature selection scores (training set)")
    save_fig(fig, "fs_01_filter_scores")

    keep = scores[~scores["low_variance"] & ~scores["redundant(|r|>0.95)"]]
    selected = list(keep.index[:k])
    log(f"  removed {len(low_var)} low-variance, {len(redundant)} redundant; kept top-{k}")
    return {"Filter (mean rank of F/MI/chi2)": selected}, scores


# --------------------------------------------------------------------------- #
# Wrapper methods
# --------------------------------------------------------------------------- #
def wrapper_methods(X: pd.DataFrame, y, cv, pool: list[str], k: int = 15, fast: bool = False):
    log("--- Wrapper methods ---")
    out = {}
    Xp = X[pool]
    est = _base_estimator()
    for direction in ("forward", "backward"):
        if fast and direction == "backward":
            continue
        t0 = time.perf_counter()
        sfs = SequentialFeatureSelector(clone(est), n_features_to_select=k, direction=direction,
                                        scoring=SCORING, cv=cv, n_jobs=config.N_JOBS)
        sfs.fit(Xp, y)
        out[f"Wrapper: SFS {direction}"] = list(Xp.columns[sfs.get_support()])
        log(f"  SFS {direction}: {k} of {len(pool)} features ({time.perf_counter() - t0:.0f}s)")

    rfe = RFE(clone(est), n_features_to_select=k, step=1).fit(Xp, y)
    out["Wrapper: RFE"] = list(Xp.columns[rfe.get_support()])

    rfecv = RFECV(clone(est), step=2, cv=cv, scoring=SCORING, min_features_to_select=3,
                  n_jobs=config.N_JOBS).fit(X, y)
    out["Wrapper: RFECV"] = list(X.columns[rfecv.get_support()])
    mean = rfecv.cv_results_["mean_test_score"]
    std = rfecv.cv_results_["std_test_score"]
    n_feats = rfecv.cv_results_.get("n_features",
                                    np.linspace(3, X.shape[1], len(mean)).astype(int))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(n_feats, mean, color="#4C72B0")
    ax.fill_between(n_feats, mean - std, mean + std, alpha=0.2)
    ax.axvline(rfecv.n_features_, ls="--", color="gray")
    ax.set_xlabel("number of features")
    ax.set_ylabel("CV balanced accuracy")
    ax.set_title(f"RFECV (logistic regression): optimum = {rfecv.n_features_} features")
    save_fig(fig, "fs_02_rfecv")
    log(f"  RFECV optimum: {rfecv.n_features_} features")
    return out


# --------------------------------------------------------------------------- #
# Embedded methods
# --------------------------------------------------------------------------- #
def embedded_methods(X: pd.DataFrame, y, k: int = 20):
    log("--- Embedded methods (L1, L2, Elastic Net, RF importance) ---")
    out, coef_tbl = {}, {}
    models = {
        "L1 (Lasso)": make_logreg("l1", C=0.1, max_iter=3000),
        "L2 (Ridge)": make_logreg("l2", C=1.0, max_iter=3000),
        "Elastic Net": make_logreg("elasticnet", C=0.1, l1_ratio=0.5, max_iter=3000),
    }
    for name, m in models.items():
        m.fit(X, y)
        imp = np.abs(m.coef_).mean(axis=0)            # mean |coef| over the 6 classes
        coef_tbl[name] = imp
        if name.startswith("L2"):
            sel = list(X.columns[np.argsort(imp)[::-1][:k]])   # L2 shrinks but never zeroes
        else:
            sel = list(X.columns[np.abs(m.coef_).max(axis=0) > 1e-6])
        out[f"Embedded: {name}"] = sel
        log(f"  {name}: {len(sel)} features")

    rf = RandomForestClassifier(400, class_weight="balanced_subsample", min_samples_leaf=2,
                                random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS).fit(X, y)
    coef_tbl["RF importance"] = rf.feature_importances_
    out["Embedded: RF importance"] = list(X.columns[np.argsort(rf.feature_importances_)[::-1][:k]])
    coef_tbl = pd.DataFrame(coef_tbl, index=X.columns)
    save_table(coef_tbl.round(5), "fs_embedded_importances")

    # Regularisation path: number of non-zero features vs C for L1 / EN
    Cs = np.logspace(-3, 1, 8)
    path = {"L1": [], "Elastic Net": []}
    for C in Cs:
        for name, pen in (("L1", "l1"), ("Elastic Net", "elasticnet")):
            m = make_logreg(pen, C=C, l1_ratio=0.5, max_iter=2000).fit(X, y)
            path[name].append(int((np.abs(m.coef_).max(axis=0) > 1e-6).sum()))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, v in path.items():
        axes[0].semilogx(Cs, v, marker="o", label=name)
    axes[0].set_xlabel("C (inverse regularisation strength)")
    axes[0].set_ylabel("non-zero features")
    axes[0].set_title("Sparsity path")
    axes[0].legend()
    top = coef_tbl["Elastic Net"].sort_values(ascending=False).head(20).index
    (coef_tbl.loc[top, ["L1 (Lasso)", "L2 (Ridge)", "Elastic Net"]]
     .plot(kind="barh", ax=axes[1]))
    axes[1].invert_yaxis()
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set_title("Mean |coefficient| (top-20 by Elastic Net)")
    save_fig(fig, "fs_03_embedded")
    return out


# --------------------------------------------------------------------------- #
# Metaheuristics
# --------------------------------------------------------------------------- #
class SubsetFitness:
    """fitness(mask) = alpha * CV_balanced_accuracy + (1 - alpha) * (1 - |S|/|F|)."""

    def __init__(self, X: pd.DataFrame, y, cv, alpha: float = 0.95):
        self.X, self.y, self.cv, self.alpha = X.values, y, cv, alpha
        self.n = X.shape[1]
        self.cache: dict[bytes, float] = {}

    def _score(self, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        acc = cross_val_score(_base_estimator(), self.X[:, mask.astype(bool)], self.y,
                              cv=self.cv, scoring=SCORING).mean()
        return self.alpha * acc + (1 - self.alpha) * (1 - mask.sum() / self.n)

    def evaluate(self, population: np.ndarray) -> np.ndarray:
        keys = [m.astype(np.uint8).tobytes() for m in population]
        todo = {k: m for k, m in zip(keys, population) if k not in self.cache}
        if len(todo) == 1:
            vals = [self._score(next(iter(todo.values())))]
        elif todo:
            vals = Parallel(n_jobs=config.N_JOBS)(delayed(self._score)(m) for m in todo.values())
        if todo:
            self.cache.update(dict(zip(todo.keys(), vals)))
        return np.array([self.cache[k] for k in keys])


def genetic_algorithm(fit: SubsetFitness, pop_size=24, n_gen=20, p_cross=0.9,
                      p_mut=None, elite=2, init_prob=0.3, seed=config.RANDOM_STATE):
    """Binary GA: tournament selection, uniform crossover, bit-flip mutation, elitism."""
    rng = np.random.default_rng(seed)
    n = fit.n
    p_mut = p_mut or 1.0 / n
    pop = (rng.random((pop_size, n)) < init_prob).astype(np.uint8)
    history = []
    for g in range(n_gen):
        f = fit.evaluate(pop)
        order = np.argsort(f)[::-1]
        history.append((f.max(), f.mean()))
        new = [pop[i].copy() for i in order[:elite]]
        while len(new) < pop_size:
            a, b = [pop[max(rng.choice(pop_size, 3, replace=False), key=lambda i: f[i])]
                    for _ in range(2)]
            if rng.random() < p_cross:
                m = rng.random(n) < 0.5
                child = np.where(m, a, b)
            else:
                child = a.copy()
            flip = rng.random(n) < p_mut
            child = np.where(flip, 1 - child, child).astype(np.uint8)
            new.append(child)
        pop = np.array(new)
    f = fit.evaluate(pop)
    return pop[np.argmax(f)].astype(bool), history


def binary_pso(fit: SubsetFitness, n_particles=24, n_iter=20, w=0.7, c1=1.5, c2=1.5,
               vmax=4.0, init_prob=0.3, seed=config.RANDOM_STATE):
    """Binary PSO (Kennedy & Eberhart 1997) with a sigmoid transfer function."""
    rng = np.random.default_rng(seed)
    n = fit.n
    X = (rng.random((n_particles, n)) < init_prob).astype(np.uint8)
    V = rng.uniform(-1, 1, (n_particles, n))
    pbest, pbest_f = X.copy(), fit.evaluate(X)
    g = pbest[np.argmax(pbest_f)].copy()
    history = []
    for _ in range(n_iter):
        r1, r2 = rng.random((2, n_particles, n))
        V = np.clip(w * V + c1 * r1 * (pbest - X) + c2 * r2 * (g - X), -vmax, vmax)
        X = (rng.random((n_particles, n)) < 1 / (1 + np.exp(-V))).astype(np.uint8)
        f = fit.evaluate(X)
        better = f > pbest_f
        pbest[better], pbest_f[better] = X[better], f[better]
        g = pbest[np.argmax(pbest_f)].copy()
        history.append((pbest_f.max(), f.mean()))
    return g.astype(bool), history


def simulated_annealing(fit: SubsetFitness, n_iter=300, T0=0.02, cooling=0.985,
                        init_prob=0.3, seed=config.RANDOM_STATE):
    """Single-solution SA: neighbour = flip 1-2 random bits."""
    rng = np.random.default_rng(seed)
    n = fit.n
    cur = (rng.random(n) < init_prob).astype(np.uint8)
    cur_f = fit.evaluate(cur[None])[0]
    best, best_f, T = cur.copy(), cur_f, T0
    history = []
    for _ in range(n_iter):
        cand = cur.copy()
        cand[rng.choice(n, rng.integers(1, 3), replace=False)] ^= 1
        cf = fit.evaluate(cand[None])[0]
        if cf > cur_f or rng.random() < np.exp((cf - cur_f) / max(T, 1e-9)):
            cur, cur_f = cand, cf
            if cf > best_f:
                best, best_f = cand.copy(), cf
        T *= cooling
        history.append((best_f, cur_f))
    return best.astype(bool), history


def metaheuristic_methods(X: pd.DataFrame, y, cv, fast: bool = False):
    log("--- Metaheuristic feature selection (GA, BPSO, SA) ---")
    fit = SubsetFitness(X, y, cv)
    size = dict(pop=10, gen=6, sa=60) if fast else dict(pop=24, gen=20, sa=300)
    runs = {}
    t0 = time.perf_counter()
    runs["GA"] = genetic_algorithm(fit, pop_size=size["pop"], n_gen=size["gen"])
    log(f"  GA done ({time.perf_counter() - t0:.0f}s, {len(fit.cache)} subsets evaluated)")
    t0 = time.perf_counter()
    runs["Binary PSO"] = binary_pso(fit, n_particles=size["pop"], n_iter=size["gen"])
    log(f"  BPSO done ({time.perf_counter() - t0:.0f}s)")
    t0 = time.perf_counter()
    runs["Simulated Annealing"] = simulated_annealing(fit, n_iter=size["sa"])
    log(f"  SA done ({time.perf_counter() - t0:.0f}s)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, (name, (_, hist)) in zip(axes, runs.items()):
        h = np.array(hist)
        ax.plot(h[:, 0], label="best", color="#C44E52")
        ax.plot(h[:, 1], label="mean / current", color="#4C72B0", alpha=0.7)
        ax.set_title(f"{name} convergence")
        ax.set_xlabel("generation / iteration")
        ax.set_ylabel("fitness")
        ax.legend()
    fig.suptitle("fitness = 0.95 * CV balanced accuracy + 0.05 * (1 - |S|/|F|)")
    save_fig(fig, "fs_04_metaheuristics_convergence")
    return {f"Metaheuristic: {k}": list(X.columns[mask]) for k, (mask, _) in runs.items()}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_feature_selection(Xtr: pd.DataFrame, ytr, groups_tr, Xte: pd.DataFrame, yte,
                          fast: bool = False):
    """Run every method; return (subsets dict, comparison table, best method name)."""
    log("=== Feature selection ===")
    cv = group_cv_splits(ytr, groups_tr, n_splits=3)

    subsets = {"All features": list(Xtr.columns)}
    filt, filter_scores = filter_methods(Xtr, ytr, k=30)
    subsets.update(filt)
    pool = [c for c in filter_scores.index
            if not filter_scores.loc[c, "low_variance"]][:25 if fast else 40]
    subsets.update(wrapper_methods(Xtr, ytr, cv, pool, k=15, fast=fast))
    subsets.update(embedded_methods(Xtr, ytr, k=20))
    subsets.update(metaheuristic_methods(Xtr, ytr, cv, fast=fast))

    rows = []
    for name, cols in subsets.items():
        if not cols:
            continue
        cv_score = cross_val_score(_base_estimator(), Xtr[cols], ytr, cv=cv, scoring=SCORING)
        clf = _base_estimator().fit(Xtr[cols], ytr)
        pred = clf.predict(Xte[cols])
        rows.append({
            "method": name, "n_features": len(cols),
            "train CV balanced acc": cv_score.mean(),
            "train CV std": cv_score.std(),
            "test accuracy": accuracy_score(yte, pred),
            "test balanced acc": balanced_accuracy_score(yte, pred),
            "test macro F1": f1_score(yte, pred, average="macro"),
        })
    comp = pd.DataFrame(rows).set_index("method").round(4)
    save_table(comp, "fs_comparison")

    # pick by inner-CV score on TRAIN (never by test); tie-break on fewer features
    best = (comp.drop(index="All features")
            .sort_values(["train CV balanced acc", "n_features"], ascending=[False, True])
            .index[0])
    log(f"  selected subset for modelling: '{best}' ({comp.loc[best, 'n_features']} features)")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [3, 1]})
    comp[["train CV balanced acc", "test balanced acc", "test macro F1"]].plot(kind="barh", ax=ax[0])
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 1)
    ax[0].set_title("Feature-selection methods (logistic regression)")
    ax[1].barh(comp.index, comp["n_features"], color="#8C8C8C")
    ax[1].invert_yaxis()
    ax[1].set_yticklabels([])
    ax[1].set_title("# features")
    save_fig(fig, "fs_05_comparison")

    # selection frequency of every feature across methods
    freq = pd.Series({c: sum(c in s for k, s in subsets.items() if k != "All features")
                      for c in Xtr.columns}).sort_values(ascending=False)
    save_table(freq.to_frame("times_selected"), "fs_selection_frequency")
    fig, ax = plt.subplots(figsize=(8, 6))
    freq.head(25).plot(kind="barh", ax=ax, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title("Most frequently selected features (across all methods)")
    ax.tick_params(axis="y", labelsize=8)
    save_fig(fig, "fs_06_selection_frequency")

    pd.Series({k: ";".join(v) for k, v in subsets.items()}).to_frame("features") \
        .to_csv(config.TABLE_DIR / "fs_selected_subsets.csv")
    return subsets, comp, best
