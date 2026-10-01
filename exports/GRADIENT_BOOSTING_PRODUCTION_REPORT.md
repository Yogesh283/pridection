# Gradient Boosting Production Report

Generated: 2026-10-01T06:10:55+00:00

## Model configuration
- PREDICTION_MODEL: **gradient_boosting**
- model_name: **gradient_boosting**
- model_version: **gb_v1**
- n_estimators: 100
- learning_rate: 0.05
- max_depth: 3
- min_history: 200
- retrain_every: 25

## Features
- count: 43
- names: lag_bs_1, lag_bs_2, lag_bs_3, lag_bs_4, lag_bs_5, lag_bs_6, lag_bs_7, lag_bs_8, lag_bs_10, lag_bs_12, lag_number_1, lag_number_2, lag_number_3, lag_number_4, lag_number_5, mean_3, mean_5, mean_8, mean_12, min_5, max_5, min_8, max_8, big_count_3, big_count_5, big_count_8, big_count_12, small_count_3, small_count_5, small_count_8, small_count_12, current_big_streak, current_small_streak, previous_streak_length, previous_color, color_lag_2, color_lag_3, color_red_count_5, color_green_count_5, color_violet_count_5, color_red_count_8, color_green_count_8, color_violet_count_8

## Training
- confirmed rounds in DB: 952
- training_rows (supervised): 932
- last_trained_at: 2026-10-01T06:10:55+00:00
- gap_detected: True (count=11999999707)

## Walk-forward smoke test
- scored: 752
- correct: 364
- incorrect: 388
- unavailable: 0
- accuracy: 48.4%
- note: chronological expanding window, retrain every 25; not a profit claim.

## Current live status
- model_status: **READY**
- decision_strategy: **gradient_boosting**
- current tip: SMALL
- confidence: 51.73%
- probability_big: 0.482703
- probability_small: 0.517297
- target_period: 20260930100050898

## Resolved live predictions (GB only)
- resolved: 0
- correct: 0
- incorrect: 0
- accuracy: n/a (insufficient sample)
- average_confidence: None
- last_10/25/50/100: None/None/None/None

## Confidence distribution
{
  "50-55": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  },
  "55-60": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  },
  "60-65": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  },
  "65-70": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  },
  "70-75": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  },
  "75+": {
    "count": 0,
    "correct": 0,
    "incorrect": 0,
    "accuracy": null
  }
}

## Errors / failures
- last_error: None
- smoke unavailable count: 0

## Retraining status
- retrain_every: 25 new confirmed rounds
- rounds_at_last_train: 952

## Production decision
- PRODUCTION CHANGE: **YES**
- PREVIOUS MODEL: majority_w8
- FINAL STRATEGY: **gradient_boosting**
- majority_w8: diagnostic only (no silent fallback)

Integrity: historical rounds and prior prediction resolutions were not modified.
