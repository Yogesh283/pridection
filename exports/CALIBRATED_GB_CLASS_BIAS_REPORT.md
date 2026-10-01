# Calibrated Gradient Boosting — Class Bias Diagnostic

Generated: 2026-10-01T08:33:43+00:00

## Scope (READ-ONLY)
- Production model: **calibrated_gradient_boosting** (frozen artifact copy).
- No production code change, no artifact overwrite, no DB mutation, no retrain-to-disk.
- Chronological test: features from rounds[:i] only; tip = BIG if P(BIG)>=0.5 else SMALL.
- Note: frozen weights were trained on later history; this measures **current model tip bias**
  on chronological features, not a pure never-seen expanding retrain OOS.

## Artifact
- path: `D:\color api\wingo30_predictor\models\artifacts\production_big_small.joblib`
- model_name: calibrated_gradient_boosting
- model_version: calibrated_gradient_boosting_20261001_1177
- trained_rows: 1127
- trained_rounds: 1177
- trained_until_period: 20261001100051004
- trained_at: 2026-10-01T08:22:36+00:00

## Historical class balance
- rounds: 1199
- actual BIG: 592 (49.37%)
- actual SMALL: 607 (50.63%)
- approx train-slice BIG%: 49.87% (n=1149)
- recent100 BIG%: 49.0%

## Frozen chronological evaluation
- test predictions: **999**
- actual BIG: 492 (49.25%)
- actual SMALL: 507 (50.75%)
- predicted BIG: 152 (15.22%)
- predicted SMALL: 847 (84.78%)
- correct BIG: 64
- incorrect BIG: 88
- correct SMALL: 419
- incorrect SMALL: 428
- BIG precision: 42.11%
- SMALL precision: 49.47%
- BIG recall: 13.01%
- SMALL recall: 82.64%
- overall accuracy: 48.35%
- wilson ~95%: (45.25, 51.45)
- pred BIG share − actual BIG share: -34.03 pp
- **class bias: SMALL-BIASED**

### Confusion matrix (actual × predicted)
```
                 pred BIG   pred SMALL
actual BIG            64          428
actual SMALL          88          419
```

### Probability stats
- avg P(BIG): 0.484303
- avg P(SMALL): 0.515697
- median P(BIG): 0.483963
- median P(SMALL): 0.516037
- min/max P(BIG): 0.428686 / 0.544948
- min/max P(SMALL): 0.455052 / 0.571314
- avg tip confidence: 0.518133

### Tip-confidence buckets
- 50-55: count=989 correct=477 incorrect=512 acc=48.23
- 55-60: count=10 correct=6 incorrect=4 acc=60.0
- 60-65: count=0 correct=0 incorrect=0 acc=None
- 65-70: count=0 correct=0 incorrect=0 acc=None
- 70+: count=0 correct=0 incorrect=0 acc=None

### Recent windows (frozen chronological)
- last 20: n=20 pred BIG/SMALL=0/20 (0.0%/100.0%) actual BIG/SMALL=9/11 acc=55.0%
- last 50: n=50 pred BIG/SMALL=3/47 (6.0%/94.0%) actual BIG/SMALL=27/23 acc=44.0%
- last 100: n=100 pred BIG/SMALL=8/92 (8.0%/92.0%) actual BIG/SMALL=49/51 acc=49.0%
- last 200: n=200 pred BIG/SMALL=22/178 (11.0%/89.0%) actual BIG/SMALL=103/97 acc=50.5%

## Live resolved predictions (DB, this model only)
- n=238
- predicted BIG/SMALL: 11/227 (4.62%/95.38%)
- actual BIG/SMALL: 120/118
- accuracy: 45.8%
- BIG/SMALL precision: 9.09% / 47.58%
- BIG/SMALL recall: 0.83% / 91.53%
- class bias: **SMALL-BIASED**
- avg P(BIG)/P(SMALL): 0.476224 / 0.523776

## Cause assessment
- Frozen chronological class_bias=SMALL-BIASED.
- Training labels near balanced (BIG=49.87%) — training class imbalance is NOT the main driver.
- Mean P(BIG)=0.484303, median=0.483963: probabilities hug 0.5; argmax threshold 0.5 amplifies tiny tilts into one-sided tip frequency.
- Overall accuracy 48.35% near chance → weak signal / underfitting; bias can dominate because there is little true predictive structure.
- 99.0% of tips sit in 50–55% confidence — model is mostly undecided; one-sided class rate is threshold artifact, not strong conviction.
- Recent100 BIG%=49.0 vs full-hist BIG%=49.37: similar — recent distribution shift is not the primary explanation.
- Live resolved DB tips for this model: pred BIG=4.62% SMALL=95.38% bias=SMALL-BIASED (n=238).
- Feature bias not proven here (no ablation). Frozen-model tip skew without matching actual-class skew indicates model/calibration/threshold behavior rather than forced 50/50 games.

## Conclusion
- Frozen chronological bias: **SMALL-BIASED**
- Live DB bias: **SMALL-BIASED**
- No thresholds were changed. No 50/50 forcing applied.
- Production remains calibrated_gradient_boosting unchanged.

