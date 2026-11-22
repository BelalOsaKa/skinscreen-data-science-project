"""Small shared helpers (saving figures/tables, timing, sklearn compatibility)."""
from __future__ import annotations

import time
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are saved, never shown
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression

from . import config

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

# Fixed, colour-blind friendly colour per class so every figure matches.
CLASS_COLORS = {
    "ACK": "#4C72B0", "BCC": "#DD8452", "MEL": "#C44E52",
    "NEV": "#55A868", "SCC": "#8172B3", "SEK": "#937860",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    log(f"{name} done in {time.perf_counter() - t0:.1f}s")


def _rel(path):
    try:
        return path.relative_to(config.PROJECT_ROOT)
    except ValueError:
        return path


def save_fig(fig, name: str) -> None:
    config.ensure_dirs()
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    log(f"  figure -> {_rel(path)}")


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    config.ensure_dirs()
    path = config.TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    log(f"  table  -> {_rel(path)}")


_SK_VERSION = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def make_logreg(penalty: str = "l2", C: float = 1.0, l1_ratio: float | None = None,
                max_iter: int = 2000, class_weight="balanced",
                random_state: int = config.RANDOM_STATE) -> LogisticRegression:
    """LogisticRegression that works across sklearn versions.

    scikit-learn >= 1.8 deprecates ``penalty`` in favour of ``l1_ratio``
    (0 = L2, 1 = L1, in between = Elastic Net).
    """
    ratio = {"l2": 0.0, "l1": 1.0}.get(penalty, l1_ratio if l1_ratio is not None else 0.5)
    solver = "lbfgs" if penalty == "l2" else "saga"
    kw = dict(C=C, solver=solver, max_iter=max_iter, class_weight=class_weight,
              random_state=random_state)
    if _SK_VERSION >= (1, 8):
        if penalty != "l2":
            kw["l1_ratio"] = ratio
    else:
        kw["penalty"] = penalty
        if penalty == "elasticnet":
            kw["l1_ratio"] = ratio
    return LogisticRegression(**kw)
