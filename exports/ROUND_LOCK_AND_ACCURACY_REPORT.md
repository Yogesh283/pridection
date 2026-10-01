# Round Lock And Accuracy Report

Generated: 2026-10-01T10:24:04+00:00

## Root cause of changing predictions
- Live runner called `generate_prediction(force=True)` every new sync tick.
- `save_prediction(update_existing=True)` **overwrote** open tip fields on each poll.
- Model retrain mid-round could flip tip probabilities before settlement.

## Fix (prediction lifecycle)
1. ROUND OPEN → confirm target_period from hist CDN
2. GENERATE PREDICTION once if no open tip exists
3. LOCK PREDICTION in DB + memory cache (immutable tip/probability/confidence)
4. WAIT FOR RESULT (1s sync updates period/settlement only)
5. ROUND SETTLED → resolve win/loss exactly once
6. ONLY THEN generate next-round prediction

## Before / after
- Before: each poll/retrain could rewrite TIP for the same target_period.
- After: first TIP for a period is permanent until settlement; polls return the same tip.

## Files changed
- `models/prediction_lock.py`
- `data/database.py` (immutable save + getters)
- `data/migrations.py` (probability_big/small, locked_at, unique key best-effort)
- `scheduler/runner.py` (lock-before-infer; heartbeat does not re-infer)
- `analysis/accuracy.py` (precision/recall/balanced + last_200)
- `tests/test_round_lock.py`
- `tools/round_lock_and_accuracy.py`

## Lock self-tests
- ROUND LOCK: **PASS**
- SAME PERIOD PREDICTION: **PASS**
- DUPLICATE PREDICTION PREVENTION: **PASS**
- PREDICTION IMMUTABILITY: **PASS**
- SETTLEMENT: **PASS**
- FUTURE DATA LEAKAGE: **PASS**
- pytest tests/test_round_lock.py: 8 passed / 0 failed

## Production artifact chronological metrics (no retrain)
- model=calibrated_gradient_boosting version=calibrated_gradient_boosting_20261001_1402 threshold=0.5
- validation: {"n": 274, "accuracy": 50.36, "balanced_accuracy": 49.13, "predicted_big_pct": 7.66, "predicted_small_pct": 92.34, "big_precision": 42.86, "small_precision": 50.99, "big_recall": 6.77, "small_recall": 91.49, "wins": 138, "losses": 136}
- holdout: {"n": 274, "accuracy": 54.74, "balanced_accuracy": 50.91, "predicted_big_pct": 2.19, "predicted_small_pct": 97.81, "big_precision": 66.67, "small_precision": 54.48, "big_recall": 3.17, "small_recall": 98.65, "wins": 150, "losses": 124, "last_20": 60.0, "last_50": 68.0, "last_100": 63.0, "last_200": 58.0}

## ExtraTrees threshold sweep (validation only — diagnostic)
- thr=0.44: acc=49.64% bal=50.72% predBIG=87.23% precB/S=48.95/54.29 recB/S=87.97/13.48
- thr=0.46: acc=51.82% bal=52.7% predBIG=79.93% precB/S=50.23/58.18 recB/S=82.71/22.7
- thr=0.48: acc=52.92% bal=53.38% predBIG=65.69% precB/S=51.11/56.38 recB/S=69.17/37.59
- thr=0.5: acc=52.55% bal=52.54% predBIG=49.27% precB/S=51.11/53.96 recB/S=51.88/53.19
- thr=0.52: acc=50.73% bal=50.14% predBIG=29.93% precB/S=48.78/51.56 recB/S=30.08/70.21
- thr=0.54: acc=52.19% bal=51.22% predBIG=16.79% precB/S=52.17/52.19 recB/S=18.05/84.4
- thr=0.56: acc=52.55% bal=51.3% predBIG=6.93% precB/S=57.89/52.16 recB/S=8.27/94.33

## Model comparison @ thr=0.48 (validation — diagnostic)
- extra_trees: acc=52.92% bal=53.38% predBIG=65.69% precB/S=51.11/56.38 recB/S=69.17/37.59
- random_forest: acc=52.55% bal=52.92% predBIG=62.41% precB/S=50.88/55.34 recB/S=65.41/40.43
- gradient_boosting: acc=48.54% bal=48.46% predBIG=47.45% precB/S=46.92/50.0 recB/S=45.86/51.06
- hist_gradient_boosting: acc=48.54% bal=48.57% predBIG=51.09% precB/S=47.14/50.0 recB/S=49.62/47.52

## Model comparison @ thr=0.48 (final holdout — diagnostic)
- extra_trees: acc=47.81% bal=49.21% predBIG=67.52% precB/S=45.41/52.81 recB/S=66.67/31.76
- random_forest: acc=48.91% bal=50.11% predBIG=64.96% precB/S=46.07/54.17 recB/S=65.08/35.14
- gradient_boosting: acc=50.36% bal=50.34% predBIG=49.64% precB/S=46.32/54.35 recB/S=50.0/50.68
- hist_gradient_boosting: acc=50.0% bal=50.24% predBIG=52.92% precB/S=46.21/54.26 recB/S=53.17/47.3

## ExtraTrees @ 0.48 untouched holdout (diagnostic)
- {"n": 274, "accuracy": 47.81, "balanced_accuracy": 49.21, "predicted_big_pct": 67.52, "predicted_small_pct": 32.48, "big_precision": 45.41, "small_precision": 52.81, "big_recall": 66.67, "small_recall": 31.76, "wins": 131, "losses": 143, "last_20": 40.0, "last_50": 32.0, "last_100": 36.0, "last_200": 46.5}

## Live resolved accuracy (settled tips only; separate from round-lock)
- overall BS: 50.59476605868358
- balanced: 50.2
- BIG precision/recall: 49.06103286384977/33.983739837398375
- SMALL precision/recall: 51.377245508982035/66.40866873065015
- last20/50/100/200: 60.0/62.0/59.0/52.5

## Production status
- Round-lock: **ACTIVATED** in runner/DB.
- Model swap: **NOT activated** by this task.
- No model auto-activated. Best diagnostic holdout in this run: gradient_boosting @0.48 = 50.36% (production remains calibrated_gradient_boosting @ 0.5).

Note: Round-locking fixes consistency, not expected accuracy.
Do not choose a model only because its prediction distribution looks balanced.

