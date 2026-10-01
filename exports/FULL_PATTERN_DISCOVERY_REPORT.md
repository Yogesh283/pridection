# Full Pattern Discovery Report

Generated: 2026-10-01T10:57:31+00:00

**READ-ONLY research. Production model/threshold/round-lock UNCHANGED.**

## 1. Dataset summary
- total_rows: 1479
- usable_rows: 1479
- duplicate_periods: 0
- missing_number: 0
- missing_color (raw missing; mapped via COLOR_MAP): 0
- missing_big_small: 0
- date/time range: {'min': '2026-09-18T01:59:40+00:00', 'max': '2026-10-01T10:52:58+00:00'}
- supervised samples (min_history=50): 1429
- split train/val/holdout: 857/285/287
- fields: big_small, color, color_raw, created_at, id, number, period, primary_color, source, timestamp

## 2–4. Number / Color / BIG-SMALL mapping (from app logic)
- Sources: analysis/statistics.py::big_small_label, config.py::COLOR_MAP, api/color_canon.py::canonical_color / primary_from_canonical
- BIG/SMALL rule: `SMALL if 0<=n<=4 else BIG`

| Number | BIG/SMALL | COLOR_MAP | Canonical | Primary |
|---:|---|---|---|---|
| 0 | SMALL | RED,VIOLET | RED,VIOLET | RED |
| 1 | SMALL | GREEN | GREEN | GREEN |
| 2 | SMALL | RED | RED | RED |
| 3 | SMALL | GREEN | GREEN | GREEN |
| 4 | SMALL | RED | RED | RED |
| 5 | BIG | GREEN,VIOLET | GREEN,VIOLET | GREEN |
| 6 | BIG | RED | RED | RED |
| 7 | BIG | GREEN | GREEN | GREEN |
| 8 | BIG | RED | RED | RED |
| 9 | BIG | GREEN | GREEN | GREEN |

## 5–8. Transition / streak / color-conditioned probs
### BIG/SMALL transitions
```json
{
  "SMALL": {
    "BIG": {
      "count": 374,
      "prob": 0.4883
    },
    "SMALL": {
      "count": 392,
      "prob": 0.5117
    }
  },
  "BIG": {
    "SMALL": {
      "count": 373,
      "prob": 0.5239
    },
    "BIG": {
      "count": 339,
      "prob": 0.4761
    }
  }
}
```
### Streak-conditioned next BS
```json
{
  "SMALL_streak_1": {
    "BIG": {
      "count": 189,
      "prob": 0.5053
    },
    "SMALL": {
      "count": 185,
      "prob": 0.4947
    }
  },
  "BIG_streak_1": {
    "SMALL": {
      "count": 191,
      "prob": 0.5121
    },
    "BIG": {
      "count": 182,
      "prob": 0.4879
    }
  },
  "SMALL_streak_2": {
    "SMALL": {
      "count": 96,
      "prob": 0.5189
    },
    "BIG": {
      "count": 89,
      "prob": 0.4811
    }
  },
  "SMALL_streak_3": {
    "BIG": {
      "count": 50,
      "prob": 0.5208
    },
    "SMALL": {
      "count": 46,
      "prob": 0.4792
    }
  },
  "BIG_streak_2": {
    "SMALL": {
      "count": 89,
      "prob": 0.489
    },
    "BIG": {
      "count": 93,
      "prob": 0.511
    }
  },
  "SMALL_streak_4": {
    "SMALL": {
      "count": 26,
      "prob": 0.5652
    },
    "BIG": {
      "count": 20,
      "prob": 0.4348
    }
  },
  "SMALL_streak_5": {
    "SMALL": {
      "count": 14,
      "prob": 0.5385
    },
    "BIG": {
      "count": 12,
      "prob": 0.4615
    }
  },
  "SMALL_streak_6": {
    "SMALL": {
      "count": 9,
      "prob": 0.6429
    },
    "BIG": {
      "count": 5,
      "prob": 0.3571
    }
  },
  "SMALL_streak_7": {
    "SMALL": {
      "count": 6,
      "prob": 0.6667
    },
    "BIG": {
      "count": 3,
      "prob": 0.3333
    }
  },
  "SMALL_streak_8": {
    "SMALL": {
      "count": 10,
      "prob": 0.625
    },
    "BIG": {
      "count": 6,
      "prob": 0.375
    }
  },
  "BIG_streak_3": {
    "SMALL": {
      "count": 56,
      "prob": 0.6022
    },
    "BIG": {
      "count": 37,
      "prob": 0.3978
    }
  },
  "BIG_streak_4": {
    "BIG": {
      "count": 17,
      "prob": 0.4595
    },
    "SMALL": {
      "count": 20,
      "prob": 0.5405
    }
  },
  "BIG_streak_5": {
    "SMALL": {
      "count": 9,
      "prob": 0.5294
    },
    "BIG": {
      "count": 8,
      "prob": 0.4706
    }
  },
  "BIG_streak_6": {
    "SMALL": {
      "count": 7,
      "prob": 0.875
    },
    "BIG": {
      "count": 1,
      "prob": 0.125
    }
  },
  "BIG_streak_7": {
    "BIG": {
      "count": 1,
      "prob": 1.0
    }
  },
  "BIG_streak_8": {
    "SMALL": {
      "count": 1,
      "prob": 1.0
    }
  }
}
```
### Prev color → next BS
```json
{
  "RED": {
    "BIG": 0.4818,
    "SMALL": 0.5182
  },
  "GREEN": {
    "SMALL": 0.517,
    "BIG": 0.483
  }
}
```
### Color transitions
```json
{
  "RED": {
    "RED": {
      "count": 374,
      "prob": 0.5047
    },
    "GREEN": {
      "count": 367,
      "prob": 0.4953
    }
  },
  "GREEN": {
    "GREEN": {
      "count": 371,
      "prob": 0.5034
    },
    "RED": {
      "count": 366,
      "prob": 0.4966
    }
  }
}
```

## 9. Concept drift (5 chronological blocks)
- Block 1: n=295 BIG%=52.2 continue_prev=48.64 P(BIG|BIG)=0.5098 P(BIG|SMALL)=0.539
- Block 2: n=295 BIG%=46.78 continue_prev=50.68 P(BIG|BIG)=0.471 P(BIG|SMALL)=0.4615
- Block 3: n=295 BIG%=49.15 continue_prev=51.36 P(BIG|BIG)=0.5034 P(BIG|SMALL)=0.4765
- Block 4: n=295 BIG%=49.49 continue_prev=47.62 P(BIG|BIG)=0.4726 P(BIG|SMALL)=0.5203
- Block 5: n=299 BIG%=43.48 continue_prev=48.66 P(BIG|BIG)=0.4109 P(BIG|SMALL)=0.4556
- Concept drift flag: **FOUND** (BIG% range=8.72, continue range=3.74)

## 10. Baselines (untouched holdout class frequencies)
- BIG/SMALL majority=56.45% random=50.0%
- NUMBER majority=13.94% random=10.0%
- COLOR majority=52.26% random=33.33%
- Continue-previous BS holdout: {'n': 287, 'accuracy': 48.43, 'balanced_accuracy': 47.56, 'small_precision': np.float64(54.32), 'big_precision': np.float64(40.8), 'small_recall': np.float64(54.32), 'big_recall': np.float64(40.8), 'f1_macro': 47.56, 'wins': 139, 'ci95': (42.71, 54.2)}
- Markov lag-1 BS holdout: {'n': 287, 'accuracy': 48.43, 'balanced_accuracy': 47.56, 'small_precision': np.float64(54.32), 'big_precision': np.float64(40.8), 'small_recall': np.float64(54.32), 'big_recall': np.float64(40.8), 'f1_macro': 47.56, 'wins': 139, 'ci95': (42.71, 54.2)}

## 11. Best single features (holdout screen, top 15)
- lag1_number: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last10_freq_4: val=50.88% hold=55.75% bal=52.12% ci=(49.96, 61.38) inv=True
- last10_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last1_mean: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last1_median: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last1_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last20_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last2_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last3_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last5_num_lag1: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- prev_number: val=52.28% hold=55.75% bal=54.86% ci=(49.96, 61.38) inv=True
- last20_num_lag16: val=51.23% hold=55.4% bal=54.65% ci=(49.62, 61.04) inv=False
- last20_freq_4: val=55.09% hold=55.05% bal=51.32% ci=(49.27, 60.7) inv=True
- last20_odd_pct: val=53.33% hold=55.05% bal=54.98% ci=(49.27, 60.7) inv=False
- last10_odd_pct: val=51.23% hold=54.01% bal=54.87% ci=(48.23, 59.68) inv=False

## 12. Model comparison — BIG/SMALL (best per row already filtered in narrative)
- Best overall: gradient_boosting / features=color / hold=55.4% / val=51.23%
- extra_trees/all: val=48.77% hold=49.13% bal=50.0% big_rec=56.8% small_rec=43.21% predBIG=56.79%
- extra_trees/bs: val=47.72% hold=48.08% bal=47.98% big_rec=47.2% small_rec=48.77% predBIG=49.48%
- extra_trees/bs_color: val=47.37% hold=49.83% bal=48.89% big_rec=41.6% small_rec=56.17% predBIG=42.86%
- extra_trees/color: val=52.98% hold=50.87% bal=49.63% big_rec=40.0% small_rec=59.26% predBIG=40.42%
- extra_trees/number: val=49.82% hold=51.57% bal=52.44% big_rec=59.2% small_rec=45.68% predBIG=56.45%
- extra_trees/number_bs: val=48.42% hold=49.48% bal=50.31% big_rec=56.8% small_rec=43.83% predBIG=56.45%
- extra_trees/number_color: val=51.58% hold=51.92% bal=53.39% big_rec=64.8% small_rec=41.98% predBIG=60.98%

## 13. NUMBER models
- Best: logistic_regression / features=number / hold=14.98% / val=9.12%
- Holdout top-2/top-3: 23.34 / 31.71

## 14. COLOR models
- Best: gradient_boosting / features=bs_color / hold=55.4% / val=44.91%

## 15. Walk-forward (ExtraTrees, BIG/SMALL)
- fold 1: n=285 acc=47.37% bal=48.17% ci=(41.65, 53.16)
- fold 2: n=285 acc=47.37% bal=47.51% ci=(41.65, 53.16)
- fold 3: n=285 acc=49.12% bal=49.14% ci=(43.37, 54.9)
- fold 4: n=285 acc=48.42% bal=48.99% ci=(42.68, 54.21)

## 16. Recent-window tests (ExtraTrees)
- window=100: hold_n≈20 acc=65.0% bal=53.3% predBIG=10.0%
- window=250: hold_n≈50 acc=46.0% bal=45.83% predBIG=50.0%
- window=500: hold_n≈100 acc=42.0% bal=44.97% predBIG=62.0%
- window=1000: hold_n≈200 acc=48.5% bal=49.47% predBIG=55.5%
- window=2000: hold_n≈286 acc=50.0% bal=50.49% predBIG=53.85%
- window=1429: hold_n≈286 acc=50.0% bal=50.49% predBIG=53.85%

## 17. Research ensemble
- Skipped — insufficient individual holdout signal to justify stacking.

## 18. Confidence
- Probabilities near 0.50–0.53 must be labeled LOW/WEAK, never HIGH. No calibrated confidence bands are activated by this research task.

## 19. Signal verdict
- NUMBER SIGNAL: **NOT_DETECTED**
  - Holdout exact 14.98% vs majority baseline 13.94% (+1.04pp). Validation was **worse** than baseline (9.12%). Not stable.
- COLOR SIGNAL: **NOT_DETECTED**
  - Holdout 55.40% vs majority 52.26% looks slightly up, but validation was **44.91%** (below baseline) and 95% CI [49.6, 61.0] overlaps majority. Unstable / likely noise.
- BIG_SMALL SIGNAL: **NOT_DETECTED**
  - Best holdout 55.40% is **below** holdout majority baseline 56.45%. Transitions ~0.48–0.52 (coin-flip). Streak rules do not survive holdout as a stable edge.
- OVERALL PREDICTIVE SIGNAL: **NOT_DETECTED**
  - No model family cleared: beat relevant baseline on **both** validation and untouched holdout with a clear margin.

## 20. Production recommendation
- **Do not activate** any research model. Production stays `calibrated_gradient_boosting` (unchanged by this task).
- Keep **round-lock** unchanged.
- Live ~0.50–0.53 BIG probabilities are weak confidence, not a tradable edge.
- Historical transitions are near 50/50; more data alone does not create predictability.
- Next useful step (only if desired later): run this same script on the **server DB** (~2.6k+ rounds) for confirmation — still read-only.

## Appendix: raw best model JSON
```json
{
  "best_bs": {
    "model": "gradient_boosting",
    "feature_mode": "color",
    "validation": {
      "n": 285,
      "accuracy": 51.23,
      "balanced_accuracy": 51.28,
      "small_precision": 50.82,
      "big_precision": 51.96,
      "small_recall": 65.49,
      "big_recall": 37.06,
      "f1_macro": 50.25,
      "wins": 146,
      "ci95": [
        45.45,
        56.98
      ]
    },
    "holdout": {
      "n": 287,
      "accuracy": 55.4,
      "balanced_accuracy": 53.55,
      "small_precision": 59.14,
      "big_precision": 48.51,
      "small_recall": 67.9,
      "big_recall": 39.2,
      "f1_macro": 53.29,
      "wins": 159,
      "ci95": [
        49.62,
        61.04
      ]
    },
    "pred_big_pct_hold": 35.19
  },
  "best_num": {
    "model": "logistic_regression",
    "feature_mode": "number",
    "validation": {
      "n": 285,
      "accuracy": 9.12,
      "ci95": [
        6.3,
        13.03
      ],
      "top_1_accuracy": 9.12,
      "top_2_accuracy": 18.95,
      "top_3_accuracy": 26.32
    },
    "holdout": {
      "n": 287,
      "accuracy": 14.98,
      "ci95": [
        11.32,
        19.57
      ],
      "top_1_accuracy": 14.98,
      "top_2_accuracy": 23.34,
      "top_3_accuracy": 31.71
    }
  },
  "best_col": {
    "model": "gradient_boosting",
    "feature_mode": "bs_color",
    "validation": {
      "n": 285,
      "accuracy": 44.91,
      "ci95": [
        39.24,
        50.72
      ],
      "top_1_accuracy": 44.91
    },
    "holdout": {
      "n": 287,
      "accuracy": 55.4,
      "ci95": [
        49.62,
        61.04
      ],
      "top_1_accuracy": 55.4
    }
  }
}
```

