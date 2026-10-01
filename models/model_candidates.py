"""Model factories shared by research and live production."""

from __future__ import annotations

import importlib.util
from typing import Any, Callable

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import TimeSeriesSplit


def candidate_factories() -> dict[str, Callable[[], Any]]:
    factories: dict[str, Callable[[], Any]] = {
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=6,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=140,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            l2_regularization=1.0,
            random_state=42,
        ),
        "calibrated_random_forest": lambda: CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=180,
                max_depth=8,
                min_samples_leaf=6,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3),
        ),
        "calibrated_gradient_boosting": lambda: CalibratedClassifierCV(
            estimator=GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.04,
                max_depth=2,
                min_samples_leaf=8,
                random_state=42,
            ),
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=3),
        ),
    }

    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        factories["xgboost"] = lambda: XGBClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    return factories


def probability_big(model: Any, row: Any) -> float:
    probabilities = model.predict_proba(row)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(1)]) if 1 in classes else 0.0
