"""Research, select, evaluate once, and activate the production BS engine."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database
from models.model_candidates import candidate_factories, probability_big
from models.production_features import FEATURE_NAMES, build_supervised_dataset

MIN_TRAINING_ROUNDS = 200
RETRAIN_EVERY = 25
THRESHOLDS = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "production_big_small.joblib"
META_PATH = ARTIFACT_DIR / "production_big_small.json"
RESULTS_PATH = ROOT / "exports" / "new_model_results.json"
REPORT_PATH = ROOT / "exports" / "NEW_MODEL_PRODUCTION_REPORT.md"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _metrics(y_true: list[int], probabilities: list[float]) -> dict[str, Any]:
    if not y_true:
        return {"predictions": 0}
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    correct = int(np.sum(pred == y))
    bins: dict[str, dict[str, Any]] = {}
    for low, high in ((0.50, 0.55), (0.55, 0.60), (0.60, 0.65),
                      (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)):
        mask = (confidence >= low) & (confidence < high)
        count = int(mask.sum())
        key = f"{low:.2f}-{min(high, 1.0):.2f}"
        bins[key] = {
            "count": count,
            "accuracy": round(float(np.mean(pred[mask] == y[mask])) * 100, 2)
            if count else None,
        }
    return {
        "predictions": len(y),
        "correct": correct,
        "incorrect": len(y) - correct,
        "accuracy": round(float(accuracy_score(y, pred)) * 100, 2),
        "precision": round(float(precision_score(y, pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y, pred, zero_division=0)) * 100, 2),
        "f1": round(float(f1_score(y, pred, zero_division=0)) * 100, 2),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y, pred)) * 100, 2
        ),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 6),
        "brier_score": round(float(brier_score_loss(y, p)), 6),
        "average_confidence": round(float(np.mean(confidence)), 4),
        "confidence_distribution": bins,
    }


def _window_metrics(y: list[int], p: list[float]) -> dict[str, Any]:
    result = {"full": _metrics(y, p)}
    for size in (10, 25, 50, 100, 200):
        result[f"last_{size}"] = _metrics(y[-size:], p[-size:])
    return result


def _walk_forward(
    factory: Callable[[], Any],
    x: np.ndarray,
    y: np.ndarray,
    target_indices: list[int],
    start: int,
    end: int,
) -> tuple[list[int], list[float]]:
    model = None
    last_fit_target = -10_000
    actuals: list[int] = []
    probabilities: list[float] = []
    indices = np.asarray(target_indices)
    for row_position, target_index in enumerate(target_indices):
        if target_index < start or target_index >= end:
            continue
        if model is None or target_index - last_fit_target >= RETRAIN_EVERY:
            train_mask = indices < target_index
            if int(train_mask.sum()) < MIN_TRAINING_ROUNDS - 50:
                continue
            model = factory()
            model.fit(x[train_mask], y[train_mask])
            last_fit_target = target_index
        probabilities.append(probability_big(model, x[row_position : row_position + 1]))
        actuals.append(int(y[row_position]))
    return actuals, probabilities


def _ensemble_probabilities(
    names: list[str],
    weights: dict[str, float],
    predictions: dict[str, tuple[list[int], list[float]]],
) -> tuple[list[int], list[float]]:
    actuals = predictions[names[0]][0]
    probs = np.zeros(len(actuals), dtype=float)
    for name in names:
        if predictions[name][0] != actuals:
            raise RuntimeError(f"unaligned predictions for {name}")
        probs += weights[name] * np.asarray(predictions[name][1], dtype=float)
    return actuals, probs.tolist()


def _learn_ensemble(
    research: dict[str, dict[str, Any]],
    validation: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, float]]:
    stable = [
        name
        for name in validation
        if research[name]["full"]["balanced_accuracy"] >= 48.0
        and validation[name]["full"]["balanced_accuracy"] >= 48.0
    ]
    stable.sort(
        key=lambda name: (
            validation[name]["full"]["balanced_accuracy"],
            -validation[name]["full"]["log_loss"],
        ),
        reverse=True,
    )
    names = stable[:4] or sorted(
        validation,
        key=lambda name: validation[name]["full"]["log_loss"],
    )[:2]
    losses = np.asarray([validation[name]["full"]["log_loss"] for name in names])
    raw = np.exp(-5.0 * (losses - losses.min()))
    raw /= raw.sum()
    return names, {name: float(weight) for name, weight in zip(names, raw)}


def _threshold_rows(y: list[int], p: list[float]) -> list[dict[str, Any]]:
    rows = []
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    pred = (p_arr >= 0.5).astype(int)
    confidence = np.maximum(p_arr, 1 - p_arr)
    for threshold in THRESHOLDS:
        mask = confidence >= threshold
        count = int(mask.sum())
        correct = int(np.sum(pred[mask] == y_arr[mask])) if count else 0
        rows.append(
            {
                "threshold": threshold,
                "predictions": count,
                "correct": correct,
                "incorrect": count - correct,
                "accuracy": round(100.0 * correct / count, 2) if count else None,
                "coverage": round(100.0 * count / len(y_arr), 2) if len(y_arr) else 0.0,
            }
        )
    return rows


def _wilson_lower(correct: int, total: int) -> float:
    if total <= 0:
        return 0.0
    z = 1.96
    p = correct / total
    denominator = 1 + z * z / total
    centre = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - margin) / denominator


def _choose_threshold(rows: list[dict[str, Any]], validation_size: int) -> float:
    minimum = max(20, int(math.ceil(validation_size * 0.10)))
    eligible = [row for row in rows if row["predictions"] >= minimum]
    if not eligible:
        return 0.55
    best = max(
        eligible,
        key=lambda row: (
            _wilson_lower(row["correct"], row["predictions"]),
            row["coverage"],
        ),
    )
    return float(best["threshold"])


def _fit_strategy(
    selected: str,
    ensemble_names: list[str],
    ensemble_weights: dict[str, float],
    factories: dict[str, Callable[[], Any]],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, Any]:
    names = ensemble_names if selected == "ensemble_v1" else [selected]
    models = {}
    for name in names:
        model = factories[name]()
        model.fit(x_train, y_train)
        models[name] = model
    return {
        "models": models,
        "weights": ensemble_weights if selected == "ensemble_v1" else {selected: 1.0},
    }


def _predict_strategy(bundle: dict[str, Any], row: np.ndarray) -> float:
    return float(
        sum(
            bundle["weights"][name] * probability_big(model, row)
            for name, model in bundle["models"].items()
        )
    )


def _walk_selected(
    selected: str,
    ensemble_names: list[str],
    ensemble_weights: dict[str, float],
    factories: dict[str, Callable[[], Any]],
    x: np.ndarray,
    y: np.ndarray,
    target_indices: list[int],
    start: int,
    end: int,
) -> tuple[list[int], list[float]]:
    indices = np.asarray(target_indices)
    fitted: dict[str, Any] | None = None
    last_fit = -10_000
    actuals: list[int] = []
    probabilities: list[float] = []
    for row_position, target_index in enumerate(target_indices):
        if target_index < start or target_index >= end:
            continue
        if fitted is None or target_index - last_fit >= RETRAIN_EVERY:
            mask = indices < target_index
            fitted = _fit_strategy(
                selected, ensemble_names, ensemble_weights, factories, x[mask], y[mask]
            )
            last_fit = target_index
        probabilities.append(
            _predict_strategy(fitted, x[row_position : row_position + 1])
        )
        actuals.append(int(y[row_position]))
    return actuals, probabilities


def _format_metric_line(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics.get('predictions', 0)} | "
        f"{metrics.get('correct', 0)} | {metrics.get('incorrect', 0)} | "
        f"{metrics.get('accuracy')}% | {metrics.get('balanced_accuracy')}% | "
        f"{metrics.get('log_loss')} | {metrics.get('brier_score')} |"
    )


def main() -> int:
    db = Database()
    rounds = db.get_rounds()
    if len(rounds) < 400:
        raise RuntimeError(f"Need at least 400 chronological rounds; found {len(rounds)}")

    x, y, target_indices = build_supervised_dataset(rounds, min_history=50)
    research_end = int(len(rounds) * 0.60)
    validation_end = int(len(rounds) * 0.80)
    factories = candidate_factories()
    research_predictions: dict[str, tuple[list[int], list[float]]] = {}
    validation_predictions: dict[str, tuple[list[int], list[float]]] = {}
    research_results: dict[str, dict[str, Any]] = {}
    validation_results: dict[str, dict[str, Any]] = {}
    training_results: dict[str, dict[str, Any]] = {}
    indices = np.asarray(target_indices)
    research_train_mask = indices < research_end

    for name, factory in factories.items():
        print(f"Evaluating {name}...")
        research_predictions[name] = _walk_forward(
            factory, x, y, target_indices, MIN_TRAINING_ROUNDS, research_end
        )
        validation_predictions[name] = _walk_forward(
            factory, x, y, target_indices, research_end, validation_end
        )
        research_results[name] = _window_metrics(*research_predictions[name])
        validation_results[name] = _window_metrics(*validation_predictions[name])
        fitted = factory()
        fitted.fit(x[research_train_mask], y[research_train_mask])
        fitted_probabilities = [
            probability_big(fitted, x[i : i + 1])
            for i in np.flatnonzero(research_train_mask)
        ]
        training_results[name] = _metrics(
            y[research_train_mask].tolist(), fitted_probabilities
        )

    ensemble_names, ensemble_weights = _learn_ensemble(
        research_results, validation_results
    )
    research_predictions["ensemble_v1"] = _ensemble_probabilities(
        ensemble_names, ensemble_weights, research_predictions
    )
    validation_predictions["ensemble_v1"] = _ensemble_probabilities(
        ensemble_names, ensemble_weights, validation_predictions
    )
    research_results["ensemble_v1"] = _window_metrics(
        *research_predictions["ensemble_v1"]
    )
    validation_results["ensemble_v1"] = _window_metrics(
        *validation_predictions["ensemble_v1"]
    )

    eligible = [
        name
        for name in validation_results
        if research_results[name]["full"]["balanced_accuracy"] >= 48.0
        and validation_results[name]["full"]["balanced_accuracy"] >= 48.0
        and (
            name == "ensemble_v1"
            or training_results[name]["accuracy"]
            - research_results[name]["full"]["accuracy"]
            <= 20.0
        )
    ]
    if not eligible:
        eligible = [
            min(
                factories,
                key=lambda name: validation_results[name]["full"]["log_loss"],
            )
        ]
    selected = max(
        eligible,
        key=lambda name: (
            validation_results[name]["full"]["balanced_accuracy"],
            validation_results[name]["full"]["accuracy"],
            -validation_results[name]["full"]["log_loss"],
        ),
    )
    selected_validation = validation_predictions[selected]
    validation_thresholds = _threshold_rows(*selected_validation)
    selected_threshold = _choose_threshold(
        validation_thresholds, len(selected_validation[0])
    )
    selected_validation_threshold = next(
        row for row in validation_thresholds
        if row["threshold"] == selected_threshold
    )
    threshold_minimum = max(
        20, int(math.ceil(len(selected_validation[0]) * 0.10))
    )
    high_confidence_enabled = bool(
        selected_validation_threshold["predictions"] >= threshold_minimum
        and selected_validation_threshold["accuracy"] is not None
        and selected_validation_threshold["accuracy"]
        >= validation_results[selected]["full"]["accuracy"]
    )

    # All choices are now frozen. The holdout is evaluated exactly once here.
    holdout_y, holdout_p = _walk_selected(
        selected,
        ensemble_names,
        ensemble_weights,
        factories,
        x,
        y,
        target_indices,
        validation_end,
        len(rounds),
    )
    holdout_results = _window_metrics(holdout_y, holdout_p)
    holdout_thresholds = _threshold_rows(holdout_y, holdout_p)

    research_fit = _fit_strategy(
        selected,
        ensemble_names,
        ensemble_weights,
        factories,
        x[research_train_mask],
        y[research_train_mask],
    )
    train_p = [
        _predict_strategy(research_fit, x[i : i + 1])
        for i in np.flatnonzero(research_train_mask)
    ]
    train_results = _metrics(y[research_train_mask].tolist(), train_p)

    # Activation happens only after the one-time holdout evaluation.
    final_fit = _fit_strategy(
        selected, ensemble_names, ensemble_weights, factories, x, y
    )
    now = datetime.now(timezone.utc)
    version = f"{selected}_{now:%Y%m%d}_{len(rounds)}"
    trained_until = str(rounds[-1]["period"])
    artifact = {
        **final_fit,
        "selected_model": selected,
        "model_name": selected,
        "model_version": version,
        "feature_names": FEATURE_NAMES,
        "feature_warmup": 50,
        "trained_rows": int(len(y)),
        "trained_rounds": len(rounds),
        "trained_until_period": trained_until,
        "trained_at": _utc_now(),
        "retrain_every": RETRAIN_EVERY,
        "confidence_threshold": selected_threshold,
        "high_confidence_enabled": high_confidence_enabled,
        "ensemble_members": ensemble_names if selected == "ensemble_v1" else [],
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    metadata = {key: value for key, value in artifact.items() if key != "models"}
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results = {
        "generated_at": _utc_now(),
        "rounds": len(rounds),
        "splits": {
            "research": [0, research_end - 1],
            "validation": [research_end, validation_end - 1],
            "holdout": [validation_end, len(rounds) - 1],
        },
        "models_available": list(factories),
        "research": research_results,
        "validation": validation_results,
        "training": training_results,
        "eligible_models": eligible,
        "ensemble_members": ensemble_names,
        "ensemble_weights": ensemble_weights,
        "selected_model": selected,
        "selected_threshold": selected_threshold,
        "high_confidence_enabled": high_confidence_enabled,
        "validation_thresholds": validation_thresholds,
        "holdout": holdout_results,
        "holdout_thresholds": holdout_thresholds,
        "train_accuracy": train_results,
        "model_version": version,
        "trained_until_period": trained_until,
        "trained_rows": len(y),
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    selected_holdout_threshold = next(
        row for row in holdout_thresholds
        if row["threshold"] == selected_threshold
    )
    rejection_lines = []
    for name in validation_results:
        if name == selected:
            continue
        reasons = []
        if research_results[name]["full"]["balanced_accuracy"] < 48.0:
            reasons.append("research walk-forward balanced accuracy below 48%")
        if validation_results[name]["full"]["balanced_accuracy"] < 48.0:
            reasons.append("validation balanced accuracy below 48%")
        if name in training_results and (
            training_results[name]["accuracy"]
            - research_results[name]["full"]["accuracy"]
            > 20.0
        ):
            reasons.append("training-to-walk-forward gap exceeded 20 points")
        if not reasons:
            reasons.append("lower validation selection score")
        rejection_lines.append(f"- `{name}`: {'; '.join(reasons)}.")
    lines = [
        "# New Model Production Report",
        "",
        f"Generated: `{results['generated_at']}`",
        "",
        "## 1. Models tested",
        "",
        ", ".join(f"`{name}`" for name in factories)
        + ". XGBoost and LightGBM were included only if already installed.",
        "",
        "## 2. Features used",
        "",
        f"{len(FEATURE_NAMES)} strictly past-only features: Big/Small lags, rolling "
        "percentages, streaks, transitions, alternating/repeat indicators; number "
        "lags 1-20, digit frequencies and rolling statistics; canonical color lags, "
        "frequencies and transitions; windows 3, 5, 8, 10, 15, 20, 30 and 50.",
        "",
        "## 3. Walk-forward methodology",
        "",
        f"Chronological 60/20/20 split over {len(rounds)} rows. Expanding-window "
        f"training begins at row {MIN_TRAINING_ROUNDS} and retrains every "
        f"{RETRAIN_EVERY} confirmed rounds. For target i, features and training use "
        "only rows before i. No random split was used.",
        "",
        "## 4. Validation results",
        "",
        "| Model | N | Correct | Incorrect | Accuracy | Balanced accuracy | Log loss | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in validation_results:
        lines.append(_format_metric_line(name, validation_results[name]["full"]))
    lines += [
        "",
        "## 5. Final holdout results",
        "",
        "The untouched holdout was evaluated once after model, weights, and threshold "
        "were frozen.",
        "",
        _format_metric_line(selected, holdout_results["full"]),
        "",
        "## 6-9. High-confidence thresholds, coverage, accuracy, counts",
        "",
        "| Threshold | Accuracy | Coverage | Predictions | Correct | Incorrect |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in holdout_thresholds:
        threshold_accuracy = (
            f"{row['accuracy']}%" if row["accuracy"] is not None else "N/A"
        )
        lines.append(
            f"| {row['threshold']:.2f} | {threshold_accuracy} | {row['coverage']}% | "
            f"{row['predictions']} | {row['correct']} | {row['incorrect']} |"
        )
    lines += [
        "",
        "Threshold selection used validation only. Rows above are holdout reporting, "
        "not threshold optimization.",
        "",
        "## 10. Calibration results",
        "",
        f"Selected holdout log loss: `{holdout_results['full']['log_loss']}`; "
        f"Brier score: `{holdout_results['full']['brier_score']}`; average confidence: "
        f"`{holdout_results['full']['average_confidence']}`. Probabilities are model "
        "outputs and are not claimed to equal observed accuracy.",
        "",
        "## 11. Selected production model",
        "",
        f"`{selected}` version `{version}`. Validation-selected confidence threshold: "
        f"`{selected_threshold:.2f}`.",
        "",
        "## 12. Why other models were rejected",
        "",
        "Models were rejected when research or validation balanced accuracy was below "
        "48%, or when training accuracy exceeded research walk-forward accuracy by "
        "more than 20 points. Among eligible models, selection used validation "
        "balanced accuracy, then accuracy, then lower log loss. Holdout results were "
        "not used for selection. Ensemble weights were derived from validation log loss.",
        "",
        *rejection_lines,
        "",
        "## 13. Leakage audit",
        "",
        "- Features for target i are built from `rounds[:i]`.",
        "- Every fit uses targets with indices strictly less than the predicted index.",
        "- No random split, future result, or historical-result modification is used.",
        "- Final holdout is excluded from feature/model/threshold/weight decisions.",
        "- `majority_w8`, last-10 follow/flip, and frequency fallback are absent from "
        "the production engine.",
        "",
        "## 14. Final live configuration",
        "",
        f"- Model: `{selected}`",
        f"- Version: `{version}`",
        f"- Artifact: `models/artifacts/{MODEL_PATH.name}`",
        f"- Trained until period: `{trained_until}`",
        f"- Trained supervised rows: `{len(y)}`",
        f"- Retrain interval: `{RETRAIN_EVERY}` confirmed rounds",
        f"- High-confidence threshold: `{selected_threshold:.2f}`",
        f"- High-confidence mode enabled by default: `{high_confidence_enabled}`",
        "- When high-confidence mode is enabled, below-threshold predictions return "
        "`PREDICTION_UNAVAILABLE`; invalid predictions always do. There is no fallback.",
        "",
        "## Overfitting check",
        "",
        f"- Train accuracy: `{train_results['accuracy']}%`",
        f"- Research walk-forward accuracy: "
        f"`{research_results[selected]['full']['accuracy']}%`",
        f"- Validation accuracy: `{validation_results[selected]['full']['accuracy']}%`",
        f"- Final holdout accuracy: `{holdout_results['full']['accuracy']}%`",
        "",
        "A holdout collapse is reported as observed; it is not hidden or used to "
        "retroactively choose another model.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    selected_walk = research_results[selected]["full"]
    print("")
    print(f"SELECTED MODEL: {selected}")
    print(f"MODEL VERSION: {version}")
    print(f"WALK-FORWARD ACCURACY: {selected_walk['accuracy']}%")
    print(f"VALIDATION ACCURACY: {validation_results[selected]['full']['accuracy']}%")
    print(f"FINAL HOLDOUT ACCURACY: {holdout_results['full']['accuracy']}%")
    print(
        "HIGH-CONFIDENCE ACCURACY: "
        f"{selected_holdout_threshold['accuracy']}%"
    )
    print(
        "HIGH-CONFIDENCE COVERAGE: "
        f"{selected_holdout_threshold['coverage']}%"
    )
    print(f"LAST 50: {holdout_results['last_50']['accuracy']}%")
    print(f"LAST 100: {holdout_results['last_100']['accuracy']}%")
    print(f"TRAINED ROWS: {len(y)}")
    print("PRODUCTION STATUS: LIVE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
