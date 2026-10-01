# New Model Production Report

Generated: `2026-10-01T06:23:00+00:00`

## 1. Models tested

`gradient_boosting`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `calibrated_random_forest`, `calibrated_gradient_boosting`. XGBoost and LightGBM were included only if already installed.

## 2. Features used

410 strictly past-only features: Big/Small lags, rolling percentages, streaks, transitions, alternating/repeat indicators; number lags 1-20, digit frequencies and rolling statistics; canonical color lags, frequencies and transitions; windows 3, 5, 8, 10, 15, 20, 30 and 50.

## 3. Walk-forward methodology

Chronological 60/20/20 split over 952 rows. Expanding-window training begins at row 200 and retrains every 25 confirmed rounds. For target i, features and training use only rows before i. No random split was used.

## 4. Validation results

| Model | N | Correct | Incorrect | Accuracy | Balanced accuracy | Log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 190 | 89 | 101 | 46.84% | 46.26% | 0.724433 | 0.265074 |
| random_forest | 190 | 89 | 101 | 46.84% | 46.33% | 0.702688 | 0.254732 |
| extra_trees | 190 | 90 | 100 | 47.37% | 46.49% | 0.703112 | 0.254883 |
| hist_gradient_boosting | 190 | 97 | 93 | 51.05% | 50.56% | 0.781835 | 0.287427 |
| calibrated_random_forest | 190 | 88 | 102 | 46.32% | 45.17% | 0.696152 | 0.251501 |
| calibrated_gradient_boosting | 190 | 97 | 93 | 51.05% | 50.29% | 0.696253 | 0.251551 |
| ensemble_v1 | 190 | 97 | 93 | 51.05% | 50.29% | 0.696253 | 0.251551 |

## 5. Final holdout results

The untouched holdout was evaluated once after model, weights, and threshold were frozen.

| calibrated_gradient_boosting | 191 | 98 | 93 | 51.31% | 50.83% | 0.695676 | 0.251257 |

Precision was 51.35%, recall 20.21%, and F1 29.01%. Chronological holdout
windows were: last 10 = 5/10 (50.00%), last 25 = 13/25 (52.00%), last 50 =
27/50 (54.00%), last 100 = 53/100 (53.00%), and full/last 200 = 98/191
(51.31%).

## 6-9. High-confidence thresholds, coverage, accuracy, counts

| Threshold | Accuracy | Coverage | Predictions | Correct | Incorrect |
|---:|---:|---:|---:|---:|---:|
| 0.55 | 18.18% | 5.76% | 11 | 2 | 9 |
| 0.60 | N/A | 0.0% | 0 | 0 | 0 |
| 0.65 | N/A | 0.0% | 0 | 0 | 0 |
| 0.70 | N/A | 0.0% | 0 | 0 | 0 |
| 0.75 | N/A | 0.0% | 0 | 0 | 0 |
| 0.80 | N/A | 0.0% | 0 | 0 | 0 |
| 0.85 | N/A | 0.0% | 0 | 0 | 0 |

Threshold selection used validation only. Rows above are holdout reporting, not threshold optimization.
The 0.55 validation subset had only 15 predictions at 46.67% accuracy, below
the minimum support rule. High-confidence filtering is available but disabled
by default; its holdout result (2/11) does not support an accuracy claim.

## 10. Calibration results

Selected holdout log loss: `0.695676`; Brier score: `0.251257`; average confidence: `0.5214`. Probabilities are model outputs and are not claimed to equal observed accuracy.

## 11. Selected production model

`calibrated_gradient_boosting` version `calibrated_gradient_boosting_20261001_952`. Validation-selected confidence threshold: `0.55`.

## 12. Why other models were rejected

Models were rejected when research or validation balanced accuracy was below 48%, or when training accuracy exceeded research walk-forward accuracy by more than 20 points. Among eligible models, selection used validation balanced accuracy, then accuracy, then lower log loss. Holdout results were not used for selection. Ensemble weights were derived from validation log loss.

- `gradient_boosting`: research walk-forward balanced accuracy below 48%; validation balanced accuracy below 48%; training-to-walk-forward gap exceeded 20 points.
- `random_forest`: research walk-forward balanced accuracy below 48%; validation balanced accuracy below 48%; training-to-walk-forward gap exceeded 20 points.
- `extra_trees`: research walk-forward balanced accuracy below 48%; validation balanced accuracy below 48%; training-to-walk-forward gap exceeded 20 points.
- `hist_gradient_boosting`: research walk-forward balanced accuracy below 48%; training-to-walk-forward gap exceeded 20 points.
- `calibrated_random_forest`: validation balanced accuracy below 48%.
- `ensemble_v1`: lower validation selection score.

## 13. Leakage audit

- Features for target i are built from `rounds[:i]`.
- Every fit uses targets with indices strictly less than the predicted index.
- No random split, future result, or historical-result modification is used.
- Final holdout is excluded from feature/model/threshold/weight decisions.
- `majority_w8`, last-10 follow/flip, and frequency fallback are absent from the production engine.

## 14. Final live configuration

- Model: `calibrated_gradient_boosting`
- Version: `calibrated_gradient_boosting_20261001_952`
- Artifact: `models/artifacts/production_big_small.joblib`
- Trained until period: `20260930100050897`
- Trained supervised rows: `902`
- Retrain interval: `25` confirmed rounds
- High-confidence threshold: `0.55`
- High-confidence mode enabled by default: `False`
- If high-confidence mode is enabled, below-threshold predictions return
  `PREDICTION_UNAVAILABLE`. Invalid predictions always return unavailable.
  There is no fallback.

## Overfitting check

- Train accuracy: `26.87%`
- Research walk-forward accuracy: `48.52%`
- Validation accuracy: `51.05%`
- Final holdout accuracy: `51.31%`

A holdout collapse is reported as observed; it is not hidden or used to retroactively choose another model.

## Production status

The new `calibrated_gradient_boosting` artifact is live as the sole Big/Small
authority. The legacy `majority_w8` path is diagnostic only and cannot be
selected through configuration.
