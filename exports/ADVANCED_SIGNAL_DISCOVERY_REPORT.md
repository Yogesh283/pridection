# Advanced Signal Discovery Report

## 1. Current baseline
- strategy: **majority_w8**
- full OOS: **49.68%** (n=618, WAIT=254, cov=70.87%)
- Wilson: [45.75, 53.61]
- blocks: [51.22, 53.23, 53.66, 46.77, 43.55]

## 2. Signals tested
- candidates_tested: **32**
- transitions o1–o4, streak continue/reverse, contexts k=2..8,
  number→BS, color→BS, majority agreement combos

## 3. Research results (top 12)
- `ctx_k6`: 60.0% n=15 cov=2.87% Δ=+10.32 status=REJECTED_SMALL_N ci=[35.75,80.18]
- `maj8_agree_col_bs`: 53.8% n=171 cov=32.7% Δ=+4.12 status=PROMISING ci=[46.33,61.11]
- `majority_w8`: 52.79% n=377 cov=72.08% Δ=+3.11 status=BASELINE ci=[47.74,57.77]
- `maj8_else_trans1`: 52.03% n=517 cov=98.85% Δ=+2.35 status=PROMISING ci=[47.73,56.31]
- `ctx_k5`: 51.89% n=185 cov=35.37% Δ=+2.21 status=UNSTABLE ci=[44.73,58.98]
- `maj8_agree_streak_rev3`: 51.89% n=264 cov=50.48% Δ=+2.21 status=PROMISING ci=[45.88,57.85]
- `streak_cont_4`: 51.75% n=371 cov=70.94% Δ=+2.07 status=PROMISING ci=[46.68,56.79]
- `streak_rev_4`: 51.21% n=371 cov=70.94% Δ=+1.53 status=UNSTABLE ci=[46.14,56.26]
- `streak_cont_3`: 51.02% n=394 cov=75.33% Δ=+1.34 status=PROMISING ci=[46.09,55.92]
- `maj8_agree_trans1`: 50.88% n=171 cov=32.7% Δ=+1.2 status=PROMISING ci=[43.45,58.27]
- `streak_rev_1`: 50.48% n=523 cov=100.0% Δ=+0.8 status=UNSTABLE ci=[46.21,54.74]
- `col_color_bs`: 50.1% n=493 cov=94.26% Δ=+0.42 status=EXPERIMENTAL ci=[45.7,54.5]

## Mutual information (lags @ research cutoff): {'lag1': 4.6e-05, 'lag2': 0.001538, 'lag3': 0.000528, 'lag4': 9e-05}

## 4. Validation results
- `majority_w8`: 45.38% n=119 Δ=-4.3 status=BASELINE blocks=[39.13, 50.0, 58.33, 37.5, 41.67]
- `maj8_agree_col_bs`: 45.16% n=62 Δ=-4.52 status=UNSTABLE blocks=[41.67, 58.33, 38.46, 41.67, 46.15]
- `maj8_else_trans1`: 43.71% n=167 Δ=-5.97 status=REJECTED blocks=[39.39, 48.48, 50.0, 42.42, 38.24]
- `maj8_agree_streak_rev3`: 50.55% n=91 Δ=+0.87 status=UNSTABLE blocks=[50.0, 55.56, 61.11, 38.89, 47.37]
- `ctx_k5`: 48.1% n=158 Δ=-1.58 status=UNSTABLE blocks=[48.39, 28.12, 41.94, 53.12, 68.75]
- `streak_cont_4`: 45.0% n=120 Δ=-4.68 status=UNSTABLE blocks=[41.67, 45.83, 58.33, 37.5, 41.67]
- `streak_rev_4`: 45.83% n=120 Δ=-3.85 status=EXPERIMENTAL blocks=[41.67, 45.83, 54.17, 41.67, 45.83]
- `streak_cont_3`: 50.41% n=123 Δ=+0.73 status=UNSTABLE blocks=[41.67, 44.0, 70.83, 44.0, 52.0]
- `maj8_agree_trans1`: 47.54% n=61 Δ=-2.14 status=UNSTABLE blocks=[41.67, 50.0, 50.0, 58.33, 38.46]

## 5. Untouched holdout
- `majority_w8`: 44.26% n=122 Δ=-5.42 ci=[35.76,53.12] blocks=[54.17, 37.5, 32.0, 58.33, 40.0]
- `maj8_agree_streak_rev3`: 46.74% n=92 Δ=-2.94 ci=[36.88,56.86] blocks=[61.11, 27.78, 42.11, 55.56, 47.37]

## 6. Best Big/Small candidates
- research (n>=80): `maj8_agree_col_bs` 53.8% n=171
- tiny research spikes rejected: ctx_k6=60.0% n=15
- validation: `maj8_agree_streak_rev3` 50.55% n=91
- holdout: `maj8_agree_streak_rev3` 46.74% n=92

## 7. Best Color candidates
- research best: `color_maj_w20` 55.49% n=474
- validation: 47.1% n=138
- holdout: 46.32% n=136
- full OOS: 52.27% n=748

## 8–10. Transition / Streak / Regime notes
- Transition/context signals near chance on research; low MI (~0) at lags 1–4.
- Streak continue/reverse did not produce stable lift over majority_w8.
- Agreement combos raise selectivity (more WAIT) without robust holdout lift.

## 11. Statistical intervals
- full majority_w8 CI [45.75, 53.61] includes 50%.

## 12. Chronological blocks
- majority_w8 full blocks: [51.22, 53.23, 53.66, 46.77, 43.55]

## 13. Rejected + reason
- All non-baseline Big/Small challengers: failed validation/holdout gate vs full baseline,
  small-N patterns, or instability.

## 14. Final production decision
- deploy: **False**
- final_strategy: **majority_w8**
- reason: Best holdout challenger maj8_agree_streak_rev3=46.74% (full baseline 49.68%, holdout control 44.26%) - reject.

Integrity fixes untouched. No historical outcomes modified.