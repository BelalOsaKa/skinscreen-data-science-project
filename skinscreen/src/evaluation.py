"""Model evaluation: 80/20 hold-out with every metric, k-fold CV, confusion
matrices, ROC-AUC, over/under-fitting diagnosis and modality comparison."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, auc, balanced_accuracy_score,
                             classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_validate, learning_curve
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .models import build_pipeline
from .preprocessing import group_cv_splits
from .utils import CLASS_COLORS, log, save_fig, save_table

CLASSES = config.CLASSES
CANCER_IDX = [CLASSES.index(c) for c in config.CANCER_CLASSES]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    tn = cm.sum() - cm.sum(axis=0) - cm.sum(axis=1) + np.diag(cm)
    fp = cm.sum(axis=0) - np.diag(cm)
    return tn / np.maximum(tn + fp, 1)


def compute_metrics(y_true, y_pred, proba=None) -> dict:
    labels = np.arange(len(CLASSES))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    m = {
        "accuracy": acc,
        "error rate": 1 - acc,
        "balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision (macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall (macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision (weighted)": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall (weighted)": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "specificity (macro)": specificity_per_class(cm).mean(),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen kappa": cohen_kappa_score(y_true, y_pred),
    }
    if proba is not None:
        try:
            m["ROC-AUC (OvR macro)"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                                     average="macro", labels=labels)
            m["ROC-AUC (OvR weighted)"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                                        average="weighted", labels=labels)
        except ValueError:
            m["ROC-AUC (OvR macro)"] = m["ROC-AUC (OvR weighted)"] = np.nan
    # secondary, clinically relevant view: cancer vs non-cancer referral
    t_c, p_c = np.isin(y_true, CANCER_IDX), np.isin(y_pred, CANCER_IDX)
    m["cancer sensitivity"] = (t_c & p_c).sum() / max(t_c.sum(), 1)
    m["cancer specificity"] = (~t_c & ~p_c).sum() / max((~t_c).sum(), 1)
    if proba is not None:
        m["cancer ROC-AUC"] = roc_auc_score(t_c, proba[:, CANCER_IDX].sum(axis=1))
    return m


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, labels=np.arange(len(CLASSES)),
                                target_names=CLASSES, output_dict=True, zero_division=0)
    tbl = pd.DataFrame(rep).T.loc[CLASSES]
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    tbl["specificity"] = specificity_per_class(cm)
    return tbl.round(4)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion(y_true, y_pred, title: str, name: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(CLASSES)))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=axes[0], cbar=False)
    axes[0].set_title("Counts")
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASSES,
                yticklabels=CLASSES, ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title("Row-normalised (recall on diagonal)")
    for a in axes:
        a.set_xlabel("predicted")
        a.set_ylabel("true")
    fig.suptitle(title)
    save_fig(fig, name)
    return cm


def plot_roc(y_true, proba, title: str, name: str) -> None:
    Y = label_binarize(y_true, classes=np.arange(len(CLASSES)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    all_fpr = np.linspace(0, 1, 200)
    tprs = []
    for i, cls in enumerate(CLASSES):
        if Y[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        tprs.append(np.interp(all_fpr, fpr, tpr))
        axes[0].plot(fpr, tpr, color=CLASS_COLORS[cls], label=f"{cls} (AUC={auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(Y.ravel(), proba.ravel())
    axes[0].plot(fpr, tpr, "k--", lw=1.2, label=f"micro-avg (AUC={auc(fpr, tpr):.3f})")
    axes[0].plot(all_fpr, np.mean(tprs, axis=0), "k:", lw=1.5,
                 label=f"macro-avg (AUC={auc(all_fpr, np.mean(tprs, axis=0)):.3f})")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=0.7)
    axes[0].set_title("One-vs-Rest ROC (6 classes)")
    axes[0].legend(fontsize=8)
    t_c = np.isin(y_true, CANCER_IDX)
    p_c = proba[:, CANCER_IDX].sum(axis=1)
    fpr, tpr, _ = roc_curve(t_c, p_c)
    axes[1].plot(fpr, tpr, color="#C44E52", label=f"cancer vs non-cancer (AUC={auc(fpr, tpr):.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", lw=0.7)
    axes[1].set_title("Referral view: P(BCC)+P(SCC)+P(MEL)")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.set_xlabel("false positive rate")
        a.set_ylabel("true positive rate")
    fig.suptitle(title)
    save_fig(fig, name)


# --------------------------------------------------------------------------- #
# Fit diagnosis
# --------------------------------------------------------------------------- #
def diagnose_fit(train_score: float, test_score: float, cv_score: float | None = None,
                 gap_tol: float = 0.10, low: float = 0.55) -> str:
    gap = train_score - test_score
    if train_score < low and test_score < low:
        return "UNDERFITTING (low train and test score: model too simple / features weak)"
    if gap > gap_tol:
        return f"OVERFITTING (train-test gap {gap:+.2f} > {gap_tol})"
    if gap > gap_tol / 2:
        return f"slight overfitting (gap {gap:+.2f})"
    return f"good fit (gap {gap:+.2f})"


def _fit(pipe, X, y, name):
    """Fit, passing balanced sample weights to models without class_weight."""
    if name == "XGBoost":
        return pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    return pipe.fit(X, y)


# --------------------------------------------------------------------------- #
# Main experiment
# --------------------------------------------------------------------------- #
def evaluate_models(models: dict, preprocessor, columns, train: pd.DataFrame, ytr,
                    test: pd.DataFrame, yte, smote: bool = False, tag: str = "") -> dict:
    """80/20 hold-out + k-fold CV (on the training portion) for every model."""
    log(f"=== Model training & evaluation {tag}===")
    cv = group_cv_splits(ytr, train[config.GROUP], n_splits=config.N_FOLDS)
    scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
               "f1_macro": "f1_macro", "precision_macro": "precision_macro",
               "recall_macro": "recall_macro", "roc_auc_ovr": "roc_auc_ovr"}

    holdout, cv_rows, fold_rows, fitted, probas = {}, [], [], {}, {}
    for name, clf in models.items():
        pipe = build_pipeline(preprocessor, columns, clone(clf), smote=smote)
        params = ({"clf__sample_weight": compute_sample_weight("balanced", ytr)}
                  if name == "XGBoost" else {})
        cvres = cross_validate(pipe, train, ytr, cv=cv, scoring=scoring,
                               return_train_score=True, n_jobs=config.N_JOBS,
                               params=params or None)
        row = {"model": name}
        for s in scoring:
            row[f"CV {s} mean"] = cvres[f"test_{s}"].mean()
            row[f"CV {s} std"] = cvres[f"test_{s}"].std()
        row["CV train accuracy mean"] = cvres["train_accuracy"].mean()
        cv_rows.append(row)
        for f, a in enumerate(cvres["test_accuracy"]):
            fold_rows.append({"model": name, "fold": f + 1, "accuracy": a,
                              "balanced_accuracy": cvres["test_balanced_accuracy"][f]})

        pipe = _fit(pipe, train, ytr, name)
        pred_te, proba_te = pipe.predict(test), pipe.predict_proba(test)
        pred_tr, proba_tr = pipe.predict(train), pipe.predict_proba(train)
        m_te, m_tr = compute_metrics(yte, pred_te, proba_te), compute_metrics(ytr, pred_tr, proba_tr)
        holdout[name] = {**{f"test {k}": v for k, v in m_te.items()},
                         **{f"train {k}": v for k, v in m_tr.items()}}
        holdout[name]["fit diagnosis"] = diagnose_fit(m_tr["balanced accuracy"],
                                                      m_te["balanced accuracy"])
        fitted[name], probas[name] = pipe, proba_te
        log(f"  {name:<24} CV acc={row['CV accuracy mean']:.3f}±{row['CV accuracy std']:.3f} "
            f"| test acc={m_te['accuracy']:.3f} bacc={m_te['balanced accuracy']:.3f} "
            f"F1={m_te['F1 (macro)']:.3f} AUC={m_te.get('ROC-AUC (OvR macro)', np.nan):.3f}")

    cv_tbl = pd.DataFrame(cv_rows).set_index("model").round(4)
    folds = pd.DataFrame(fold_rows)
    hold = pd.DataFrame(holdout).T
    num_cols = [c for c in hold.columns if c != "fit diagnosis"]
    hold[num_cols] = hold[num_cols].astype(float).round(4)
    save_table(cv_tbl, f"eval_kfold_cv{tag}")
    save_table(folds, f"eval_kfold_per_fold{tag}", index=False)
    save_table(hold, f"eval_holdout_all_metrics{tag}")

    _plot_model_comparison(cv_tbl, hold, tag)
    _plot_cv_folds(folds, tag)

    # ---- detailed analysis of the best model (by CV, not by test) --------- #
    best = cv_tbl["CV balanced_accuracy mean"].idxmax()
    log(f"  best model by CV balanced accuracy: {best}")
    pipe = fitted[best]
    pred = pipe.predict(test)
    cm = plot_confusion(yte, pred, f"{best}: confusion matrix (20% patient-level test set)",
                        f"eval_01_confusion_matrix_best{tag}")
    plot_roc(yte, probas[best], f"{best}: ROC curves (test set)", f"eval_02_roc_best{tag}")
    pc = per_class_table(yte, pred)
    save_table(pc, f"eval_per_class_best{tag}")

    # confusion matrices for all models (small multiples)
    k = len(fitted)
    ncol = 4
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow))
    for ax, (name, p) in zip(axes.ravel(), fitted.items()):
        c = confusion_matrix(yte, p.predict(test), labels=np.arange(len(CLASSES)))
        sns.heatmap(c / np.maximum(c.sum(1, keepdims=True), 1), annot=True, fmt=".2f",
                    cmap="Blues", cbar=False, xticklabels=CLASSES, yticklabels=CLASSES,
                    ax=ax, vmin=0, vmax=1, annot_kws={"size": 7})
        ax.set_title(name, fontsize=10)
    for ax in axes.ravel()[k:]:
        ax.axis("off")
    fig.suptitle("Row-normalised confusion matrices, all models (test set)")
    save_fig(fig, f"eval_03_confusion_all_models{tag}")

    learning_curves(pipe, train, ytr, cv, best, tag)

    return {"cv": cv_tbl, "holdout": hold, "folds": folds, "best": best,
            "best_cm": cm, "per_class": pc, "fitted": fitted, "probas": probas}


def _plot_model_comparison(cv_tbl, hold, tag):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    order = cv_tbl["CV balanced_accuracy mean"].sort_values().index
    y = np.arange(len(order))
    for off, col, lab in ((-0.2, "CV accuracy mean", "accuracy"),
                          (0.0, "CV balanced_accuracy mean", "balanced acc"),
                          (0.2, "CV f1_macro mean", "macro F1")):
        axes[0].barh(y + off, cv_tbl.loc[order, col], height=0.2,
                     xerr=cv_tbl.loc[order, col.replace("mean", "std")], label=lab)
    axes[0].set_yticks(y, order)
    axes[0].set_xlim(0, 1)
    axes[0].set_title(f"{config.N_FOLDS}-fold patient-grouped CV (mean ± std)")
    axes[0].legend(loc="lower right")

    tr, te = hold.loc[order, "train balanced accuracy"], hold.loc[order, "test balanced accuracy"]
    axes[1].barh(y - 0.2, tr, height=0.4, label="train", color="#8C8C8C")
    axes[1].barh(y + 0.2, te, height=0.4, label="test", color="#4C72B0")
    for i, (a, b) in enumerate(zip(tr, te)):
        axes[1].text(max(a, b) + 0.01, i, f"gap {a - b:+.2f}", va="center", fontsize=8)
    axes[1].set_yticks(y, [])
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("Over/under-fitting: train vs test balanced accuracy")
    axes[1].legend(loc="lower right")
    save_fig(fig, f"eval_04_model_comparison{tag}")


def _plot_cv_folds(folds, tag):
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.boxplot(data=folds, x="model", y="accuracy", ax=ax, color="#A1C9F4")
    sns.stripplot(data=folds, x="model", y="accuracy", ax=ax, color="k", size=4)
    ax.tick_params(axis="x", rotation=30)
    ax.set_title(f"Accuracy per fold ({config.N_FOLDS}-fold CV)")
    save_fig(fig, f"eval_05_cv_folds{tag}")


def learning_curves(pipe, X, y, cv, name, tag=""):
    sizes, tr, va = learning_curve(clone(pipe), X, y, cv=cv, scoring="balanced_accuracy",
                                   train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=config.N_JOBS,
                                   shuffle=True, random_state=config.RANDOM_STATE,
                                   error_score=np.nan)  # tiny subsets can break SMOTE
    ok = ~np.isnan(va).all(axis=1)
    sizes, tr, va = sizes[ok], tr[ok], va[ok]
    tr, va = np.ma.masked_invalid(tr), np.ma.masked_invalid(va)
    fig, ax = plt.subplots(figsize=(7, 4))
    for s, lab, c in ((tr, "training", "#8C8C8C"), (va, "validation", "#4C72B0")):
        ax.plot(sizes, s.mean(1), marker="o", color=c, label=lab)
        ax.fill_between(sizes, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=c, alpha=0.15)
    ax.set_xlabel("training samples")
    ax.set_ylabel("balanced accuracy")
    ax.set_title(f"Learning curve: {name}")
    ax.legend()
    save_fig(fig, f"eval_06_learning_curve{tag}")


# --------------------------------------------------------------------------- #
# Modality comparison (tabular vs image vs early/late fusion)
# --------------------------------------------------------------------------- #
def modality_comparison(clf, make_preprocessor, train, ytr, test, yte,
                        selected_cols, name: str) -> pd.DataFrame:
    log(f"=== Modality comparison with {name} ===")
    rows, probs = {}, {}
    for label, (tab, img) in {"Clinical data only": (True, False),
                              "Image features only": (False, True),
                              "Early fusion (all features)": (True, True)}.items():
        prep = make_preprocessor(tab, img)
        cols = list(prep.fit(train).get_feature_names_out())
        pipe = _fit(build_pipeline(prep, cols, clone(clf)), train, ytr, name)
        probs[label] = pipe.predict_proba(test)
        rows[label] = compute_metrics(yte, probs[label].argmax(1), probs[label])
    # early fusion + feature selection
    prep = make_preprocessor(True, True)
    pipe = _fit(build_pipeline(prep, selected_cols, clone(clf)), train, ytr, name)
    p = pipe.predict_proba(test)
    rows["Early fusion + feature selection"] = compute_metrics(yte, p.argmax(1), p)
    # late fusion: average the probabilities of the two single-modality models
    p = (probs["Clinical data only"] + probs["Image features only"]) / 2
    rows["Late fusion (mean probabilities)"] = compute_metrics(yte, p.argmax(1), p)

    tbl = pd.DataFrame(rows).T.round(4)
    save_table(tbl, "eval_modality_comparison")
    cols = ["accuracy", "balanced accuracy", "F1 (macro)", "ROC-AUC (OvR macro)", "cancer sensitivity"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    tbl[cols].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)
    ax.set_title(f"Clinical vs image vs fusion ({name}, test set)")
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, "eval_07_modality_comparison")
    return tbl
