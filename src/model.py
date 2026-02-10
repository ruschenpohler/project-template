"""Model training module.

Called by run_experiment.py. Receives the validated experiment config and a
ShadowDataset, trains the specified model, evaluates it, and returns a metrics
dict that run_experiment.py logs to MLflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.config_schema import ExperimentConfig
from src.data_access import ShadowDataset

TARGET_COLUMN = "target"

# Maps evaluation.metric names to sklearn scoring functions.
# All scorers follow signature (y_true, y_pred_or_proba) -> float.
_METRIC_FUNCTIONS: dict[str, Any] = {
    "roc_auc": lambda y, prob: roc_auc_score(y, prob),
    "f1": lambda y, prob: f1_score(y, (prob >= 0.5).astype(int)),
    "precision": lambda y, prob: precision_score(y, (prob >= 0.5).astype(int)),
    "recall": lambda y, prob: recall_score(y, (prob >= 0.5).astype(int)),
    "log_loss": lambda y, prob: log_loss(y, prob),
}


def _build_model(
    model_type: str, hyperparameters: dict[str, Any], random_seed: int
) -> Any:
    """Instantiate the model specified by config."""
    if model_type == "xgboost":
        return XGBClassifier(
            random_state=random_seed,
            eval_metric="logloss",
            **hyperparameters,
        )
    if model_type == "logistic_regression":
        return LogisticRegression(
            random_state=random_seed,
            **hyperparameters,
        )
    if model_type == "random_forest":
        return RandomForestClassifier(
            random_state=random_seed,
            **hyperparameters,
        )
    # lightgbm is an allowed model type but not a Phase 1 dependency.
    # Add it when lightgbm is added to pyproject.toml.
    raise ValueError(
        f"Model type '{model_type}' is not yet implemented. "
        f"Currently supported: xgboost, logistic_regression, random_forest."
    )


def train_and_evaluate(
    config: ExperimentConfig, dataset: ShadowDataset
) -> dict[str, float]:
    """Train a model and return evaluation metrics.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration.
    dataset : ShadowDataset
        Dataset loaded via src/data_access.py.

    Returns
    -------
    dict[str, float]
        All computed metrics keyed by metric name
        (e.g. {"roc_auc": 0.85, "f1": 0.78, ...}).
    """
    df = dataset.df
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    model = _build_model(config.model_type, config.hyperparameters, config.random_seed)

    # --- split strategy ---------------------------------------------------
    if config.evaluation.split_strategy == "stratified_kfold":
        skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.random_seed)
        fold_probas = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]
            fold_model = _build_model(
                config.model_type, config.hyperparameters, config.random_seed
            )
            fold_model.fit(X_train, y_train)
            fold_probas[val_idx] = fold_model.predict_proba(X_val)[:, 1]

        y_proba = fold_probas

    elif config.evaluation.split_strategy == "temporal_split":
        # For temporal split, use the last 30% as test (preserving row order).
        split_idx = int(len(df) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y = y_test
    else:
        raise ValueError(
            f"Unknown split_strategy '{config.evaluation.split_strategy}'"
        )

    # --- compute all metrics ----------------------------------------------
    metrics: dict[str, float] = {}
    for name, fn in _METRIC_FUNCTIONS.items():
        try:
            metrics[name] = float(fn(y, y_proba))
        except Exception:
            # Some metrics may fail on tiny/degenerate data (e.g. single class
            # in a fold). Record NaN rather than crashing the whole run.
            metrics[name] = float("nan")

    return metrics
