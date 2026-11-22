"""Classifier zoo and full (leakage-free) training pipelines."""
from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config
from .preprocessing import ColumnNameSelector
from .utils import make_logreg

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:  # pragma: no cover
    HAS_IMBLEARN = False


def get_models(fast: bool = False) -> dict:
    """Classifiers covering linear, distance-based, probabilistic, tree,
    ensemble, kernel and neural models. Class imbalance is handled with
    class weights where the algorithm supports it."""
    rs = config.RANDOM_STATE
    n_trees = 150 if fast else 500
    models = {
        "Logistic Regression": make_logreg("l2", C=0.5),
        "LDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "Gaussian NB": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=5,
                                                class_weight="balanced", random_state=rs),
        "Random Forest": RandomForestClassifier(n_trees, min_samples_leaf=2,
                                                class_weight="balanced_subsample",
                                                random_state=rs, n_jobs=config.N_JOBS),
        "Extra Trees": ExtraTreesClassifier(n_trees, min_samples_leaf=2,
                                            class_weight="balanced_subsample",
                                            random_state=rs, n_jobs=config.N_JOBS),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=150 if fast else 300, max_leaf_nodes=15,
            l2_regularization=1.0, class_weight="balanced", random_state=rs),
        "SVM (RBF)": SVC(C=3.0, kernel="rbf", gamma="scale", probability=True,
                         class_weight="balanced", random_state=rs),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=400,
                             early_stopping=True, random_state=rs),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200 if fast else 400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0,
            eval_metric="mlogloss", random_state=rs, n_jobs=config.N_JOBS)
    return models


def build_pipeline(preprocessor, columns, clf, smote: bool = False):
    """preprocess -> keep selected columns -> (SMOTE) -> classifier.

    Everything is inside one Pipeline so that, in cross-validation, the
    imputers/scalers/oversampler are re-fitted on each training fold only.
    """
    from sklearn.base import clone

    steps = [("prep", clone(preprocessor)), ("select", ColumnNameSelector(columns))]
    if smote:
        if not HAS_IMBLEARN:
            raise ImportError("--smote needs `pip install imbalanced-learn`")
        steps.append(("smote", SMOTE(k_neighbors=3, random_state=config.RANDOM_STATE)))
        steps.append(("clf", clf))
        return ImbPipeline(steps)
    steps.append(("clf", clf))
    return Pipeline(steps)
