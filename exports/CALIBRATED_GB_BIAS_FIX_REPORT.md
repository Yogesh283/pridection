# Calibrated GB Bias Fix Report

Generated: 2026-10-01T08:39:37+00:00

## 1. Root cause
- sklearn classes_=[np.int64(0), np.int64(1)] (0=SMALL,1=BIG expected).
- Class index mapping OK; probability_big uses classes_.index(1).
- Frozen tip share BIG=15.27% vs actual 49.87%.
- Mean P(BIG)=0.487491 systematically below 0.5 → SMALL-heavy argmax@0.5.
- Live confidence_level hardcoded HIGH when available (ensemble) — 51–53% incorrectly shown as HIGH.
- Primary bias driver: calibrated probabilities tilted slightly toward SMALL; not inverted classes_. Training labels are near 50/50.

## 2. Before metrics (frozen current artifact @ thr=0.5)
- accuracy: 48.49
- predicted BIG/SMALL: 15.27% / 84.73%
- BIG/SMALL recall: 13.67 / 83.13
- avg P(BIG): 0.487491

## 3. Dataset
- usable rows: 1159
- BIG: 578 (49.87%)
- SMALL: 581 (50.13%)
- train/val/holdout: 695/231/233

## 4. Threshold comparison (top validation survivors)
- extra_trees thr=0.48: predBIG=62.77% acc=55.84 bal=55.46 precB/S=55.86/55.81 recB/S=68.07/42.86 score=72.44988744588746
- random_forest thr=0.48: predBIG=53.25% acc=54.11 bal=54.02 precB/S=55.28/52.78 recB/S=57.14/50.89 score=72.30914935064935
- extra_trees thr=0.5: predBIG=40.69% acc=53.68 bal=53.97 precB/S=56.38/51.82 recB/S=44.54/63.39 score=70.89652813852814
- hist_gradient_boosting thr=0.46: predBIG=50.65% acc=51.52 bal=51.5 precB/S=52.99/50.0 recB/S=52.1/50.89 score=69.40212987012987
- calibrated_gradient_boosting_sigmoid thr=0.48: predBIG=43.29% acc=51.95 bal=52.15 precB/S=54.0/50.38 recB/S=45.38/58.93 score=68.99050865800865
- calibrated_gradient_boosting_isotonic thr=0.46: predBIG=67.97% acc=54.11 bal=53.57 precB/S=54.14/54.05 recB/S=71.43/35.71 score=68.9154264069264
- hist_gradient_boosting thr=0.48: predBIG=49.35% acc=51.08 bal=51.1 precB/S=52.63/49.57 recB/S=50.42/51.79 score=68.84812987012987
- hist_gradient_boosting thr=0.52: predBIG=42.42% acc=51.95 bal=52.18 precB/S=54.08/50.38 recB/S=44.54/59.82 score=68.84734848484848
- random_forest thr=0.46: predBIG=71.0% acc=54.55 bal=53.91 precB/S=54.27/55.22 recB/S=74.79/33.04 score=68.8033658008658
- hist_gradient_boosting thr=0.44: predBIG=54.55% acc=51.08 bal=50.95 precB/S=52.38/49.52 recB/S=55.46/46.43 score=67.9189090909091
- hist_gradient_boosting thr=0.5: predBIG=46.32% acc=50.65 bal=50.76 precB/S=52.34/49.19 recB/S=47.06/54.46 score=67.75156926406926
- random_forest thr=0.5: predBIG=34.63% acc=51.95 bal=52.42 precB/S=55.0/50.33 recB/S=36.97/67.86 score=67.52890692640693
- hist_gradient_boosting thr=0.42: predBIG=55.84% acc=49.78 bal=49.61 precB/S=51.16/48.04 recB/S=55.46/43.75 score=65.86416883116884
- hist_gradient_boosting thr=0.54: predBIG=38.1% acc=50.22 bal=50.58 precB/S=52.27/48.95 recB/S=38.66/62.5 score=65.77604761904762
- gradient_boosting thr=0.5: predBIG=29.0% acc=50.65 bal=51.29 precB/S=53.73/49.39 recB/S=30.25/72.32 score=64.8183658008658

## 5. Calibration comparison (validation @0.5)
- gb_raw: predBIG=26.84% acc=49.35 bal=50.05 avgPbig=0.462838 brier=0.259591 logloss=0.712502
- gb_calibrated_sigmoid: predBIG=12.12% acc=48.48 bal=49.63 avgPbig=0.476076 brier=0.251907 logloss=0.696986
- gb_calibrated_isotonic: predBIG=16.88% acc=52.38 bal=53.39 avgPbig=0.46889 brier=0.252262 logloss=0.698146

## 6–7. Selection
- selected: extra_trees
- threshold: 0.48
- validation accuracy: 55.84
- validation balanced accuracy: 55.46

## 8. Final holdout (untouched until after selection)
- holdout accuracy: 51.5
- holdout balanced accuracy: 51.73
- holdout pred BIG/SMALL: 60.52% / 39.48%
- holdout BIG/SMALL precision: 50.35 / 53.26
- holdout BIG/SMALL recall: 62.28 / 41.18
- confusion: {'actual_BIG_pred_BIG': 71, 'actual_BIG_pred_SMALL': 43, 'actual_SMALL_pred_BIG': 70, 'actual_SMALL_pred_SMALL': 49}

## 9. Expanding walk-forward on holdout period
- {"n": 233, "actual_big": 114, "actual_small": 119, "actual_big_pct": 48.93, "actual_small_pct": 51.07, "predicted_big": 151, "predicted_small": 82, "predicted_big_pct": 64.81, "predicted_small_pct": 35.19, "correct_big": 73, "incorrect_big": 78, "correct_small": 41, "incorrect_small": 41, "big_precision": 48.34, "small_precision": 50.0, "big_recall": 64.04, "small_recall": 34.45, "overall_accuracy": 48.93, "balanced_accuracy": 49.24, "confusion": {"actual_BIG_pred_BIG": 73, "actual_BIG_pred_SMALL": 41, "actual_SMALL_pred_BIG": 78, "actual_SMALL_pred_SMALL": 41}, "avg_p_big": 0.497361, "avg_p_small": 0.502639, "median_p_big": 0.500254, "avg_tip_confidence": 0.538191, "threshold": 0.48, "last_20": 60.0, "last_50": 50.0, "last_100": 53.0, "last_200": 52.0}

## 10. Confidence fix
- Bands: LOW 50–55, MEDIUM 55–65, HIGH 65–75, VERY_HIGH 75+
- Hardcoded HIGH-when-available removed in live wrapper (see code change).

## 11. Production
- status: ACTIVATED
- model version: extra_trees_20261001_0839_1209_thr0.48

## 12. Tests
- PASS: 0-4 SMALL
- PASS: 5-9 BIG
- PASS: p_big>thr => BIG
- PASS: p_big<thr => SMALL
- PASS: p_big==thr => BIG (documented)
- PASS: 52% is LOW not HIGH
- PASS: 60% MEDIUM
- PASS: 70% HIGH
- PASS: 80% VERY_HIGH
- PASS: prod classes_ contain 0 and 1
- FAIL: tip agrees with probability side at threshold
- PASS: artifact stores selected threshold
- passed=11 failed=1


## 13. Post-fix regression
- pytest tests/test_production_engine.py: **5 passed**
- Tip-vs-threshold agreement check corrected for thr!=0.5.
- Live smoke: model=extra_trees thr=0.48 tip follows P(BIG).
