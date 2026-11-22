"""Tabular + image preprocessing pipeline and leakage-free (patient-level) splits."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .utils import log


class IQRClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to [Q1 - k*IQR, Q3 + k*IQR] learnt on the training data.

    Reduces the influence of noisy measurements (e.g. mistyped lesion
    diameters) without throwing samples away.
    """

    def __init__(self, k: float = 3.0):
        self.k = k

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.k * iqr, q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)


class ColumnNameSelector(BaseEstimator, TransformerMixin):
    """Keep a fixed list of columns of a pandas DataFrame (used after feature selection)."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[list(self.columns)] if self.columns is not None else X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns if self.columns is not None else input_features)


def build_preprocessor(df: pd.DataFrame, image_cols: list[str], indicator_cols: list[str],
                       include_tabular: bool = True, include_image: bool = True,
                       imputer: str = "median") -> ColumnTransformer:
    """ColumnTransformer producing a pandas DataFrame with readable feature names.

    * numeric  : IQR clipping -> median (or KNN) imputation -> standardisation
    * categor. : 'MISSING' constant imputation -> one-hot encoding
                 (vocabulary fixed from the full table so every fold has the
                 same columns; this uses no label information)
    * binary   : most-frequent imputation (missingness kept via *_missing cols)
    * image    : median imputation -> standardisation
    """
    transformers = []
    if include_tabular:
        num_imp = KNNImputer(n_neighbors=5) if imputer == "knn" else SimpleImputer(strategy="median")
        transformers.append(("num", Pipeline([
            ("clip", IQRClipper(k=3.0)),
            ("impute", num_imp),
            ("scale", StandardScaler()),
        ]), config.NUMERIC_COLS))

        categories = [sorted(df[c].fillna("MISSING").unique().tolist())
                      for c in config.CATEGORICAL_COLS]
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(categories=categories, handle_unknown="ignore",
                                     sparse_output=False)),
        ]), config.CATEGORICAL_COLS))

        transformers.append(("bin", SimpleImputer(strategy="most_frequent"),
                             config.BINARY_COLS))
        if indicator_cols:
            transformers.append(("miss", "passthrough", indicator_cols))

    if include_image and image_cols:
        transformers.append(("img", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), image_cols))

    ct = ColumnTransformer(transformers, remainder="drop",
                           verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def patient_split(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
                  seed: int = config.RANDOM_STATE):
    """Stratified 80/20 split where no patient appears in both train and test.

    One patient can have several images/lesions; a random image split would
    leak near-duplicate photos into the test set and inflate the scores.
    """
    n_splits = int(round(1 / test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[config.TARGET], df[config.GROUP]))
    tr, te = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    overlap = set(tr[config.GROUP]) & set(te[config.GROUP])
    assert not overlap, f"patient leakage between splits: {list(overlap)[:5]}"
    log(f"Patient-level split: train={len(tr)} ({len(tr)/len(df):.0%}) "
        f"test={len(te)} ({len(te)/len(df):.0%}), no shared patients")
    return tr, te


def group_cv_splits(y, groups, n_splits: int = config.N_FOLDS,
                    seed: int = config.RANDOM_STATE) -> list:
    """Pre-computed StratifiedGroupKFold splits (re-usable list, for cv=...)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(sgkf.split(np.zeros(len(y)), y, groups))


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    s = pd.DataFrame({
        "train": train[config.TARGET].value_counts(),
        "test": test[config.TARGET].value_counts(),
    }).reindex(config.CLASSES).fillna(0).astype(int)
    s["train_%"] = (100 * s["train"] / s["train"].sum()).round(1)
    s["test_%"] = (100 * s["test"] / s["test"].sum()).round(1)
    return s
