# Color Signal Deep Research Report

Generated: 2026-10-01T11:48:32+00:00

**READ-ONLY. Production UNCHANGED. No model activation.**

## Banner: COLOR SIGNAL NOT STABLE

## 1. Color mapping (from application code)
- Sources: config.py::COLOR_MAP, api/color_canon.py::canonical_color / primary_from_canonical, analysis/statistics.py::big_small_label

| Number | COLOR_MAP | Canonical | Primary | BIG/SMALL |
|---:|---|---|---|---|
| 0 | RED,VIOLET | RED,VIOLET | RED | SMALL |
| 1 | GREEN | GREEN | GREEN | SMALL |
| 2 | RED | RED | RED | SMALL |
| 3 | GREEN | GREEN | GREEN | SMALL |
| 4 | RED | RED | RED | SMALL |
| 5 | GREEN,VIOLET | GREEN,VIOLET | GREEN | BIG |
| 6 | RED | RED | RED | BIG |
| 7 | GREEN | GREEN | GREEN | BIG |
| 8 | RED | RED | RED | BIG |
| 9 | GREEN | GREEN | GREEN | BIG |

- Historical primary consistency checks: 1580 mismatches=0

## 2. Dataset audit
- total_rounds: 1580
- usable_rounds: 1580
- duplicates: 0
- missing_numbers: 0
- missing_colors: 0
- missing_big_small: 0
- range: {'min': '2026-09-18T01:59:40+00:00', 'max': '2026-10-01T11:43:27+00:00', 'period_min': '20260918100050239', 'period_max': '20261001100051407'}

## 3. Color baseline
- distribution: {"RED": {"count": 790, "pct": 50.0}, "GREEN": {"count": 790, "pct": 50.0}, "VIOLET": {"count": 0, "pct": 0.0}}
- majority_color: RED
- majority_baseline: 50.0%
- random_baseline: 33.33%

## 4. Color transitions
```json
{
  "transitions": {
    "RED": {
      "RED": {
        "count": 397,
        "prob": 0.5025
      },
      "GREEN": {
        "count": 393,
        "prob": 0.4975
      },
      "VIOLET": {
        "count": 0,
        "prob": 0.0
      },
      "_n": 790
    },
    "GREEN": {
      "RED": {
        "count": 392,
        "prob": 0.4968
      },
      "GREEN": {
        "count": 397,
        "prob": 0.5032
      },
      "VIOLET": {
        "count": 0,
        "prob": 0.0
      },
      "_n": 789
    },
    "VIOLET": {
      "RED": {
        "count": 0,
        "prob": null
      },
      "GREEN": {
        "count": 0,
        "prob": null
      },
      "VIOLET": {
        "count": 0,
        "prob": null
      },
      "_n": 0
    }
  },
  "in_sample_markov_acc": 50.28
}
```

## 5. Streak analysis
```json
{
  "by_streak": {
    "1": {
      "n": 785,
      "dist": {
        "RED": {
          "count": 402,
          "pct": 51.21
        },
        "GREEN": {
          "count": 383,
          "pct": 48.79
        }
      },
      "continue_rate_if_same_as_prev_overall": null
    },
    "2": {
      "n": 396,
      "dist": {
        "GREEN": {
          "count": 208,
          "pct": 52.53
        },
        "RED": {
          "count": 188,
          "pct": 47.47
        }
      },
      "continue_rate_if_same_as_prev_overall": null
    },
    "3": {
      "n": 195,
      "dist": {
        "GREEN": {
          "count": 97,
          "pct": 49.74
        },
        "RED": {
          "count": 98,
          "pct": 50.26
        }
      },
      "continue_rate_if_same_as_prev_overall": null
    },
    "4": {
      "n": 100,
      "dist": {
        "GREEN": {
          "count": 50,
          "pct": 50.0
        },
        "RED": {
          "count": 50,
          "pct": 50.0
        }
      },
      "continue_rate_if_same_as_prev_overall": null
    },
    ">=5": {
      "n": 103,
      "dist": {
        "RED": {
          "count": 51,
          "pct": 49.51
        },
        "GREEN": {
          "count": 52,
          "pct": 50.49
        }
      },
      "continue_rate_if_same_as_prev_overall": null
    }
  },
  "continue_prev_rate": 50.28,
  "change_rate": 49.72,
  "n": 1579
}
```

## 6. Feature ablation (best holdout improvement per group)
- previous_color: best_improvement=-0.0 flag=NOT_DETECTED
- color_history: best_improvement=5.56 flag=FOUND
- color_transition: best_improvement=-0.98 flag=NOT_DETECTED
- color_frequency: best_improvement=5.89 flag=FOUND
- color_only: best_improvement=1.31 flag=WEAK
- color_bs: best_improvement=4.25 flag=NOT_DETECTED
- color_number: best_improvement=0.66 flag=WEAK
- color_bs_number: best_improvement=3.27 flag=FOUND

## 7. Rolling window study
- feature/model locked from val selection: color_transition / logistic_regression
- window=50: n=306 acc=49.02% bal=49.37% ci=(43.47, 54.6)
- window=100: n=306 acc=51.96% bal=51.96% ci=(46.37, 57.5)
- window=200: n=306 acc=45.75% bal=45.82% ci=(40.26, 51.35)
- window=300: n=306 acc=43.79% bal=43.82% ci=(38.34, 49.39)
- window=500: n=306 acc=46.41% bal=46.43% ci=(40.9, 52.0)
- window=1000: n=306 acc=49.02% bal=49.59% ci=(43.47, 54.6)
- window=all: n=306 acc=50.0% bal=50.21% ci=(44.43, 55.57)

## 8. Championship table
- REJECTED: logistic_regression | color_transition | val=55.23 hold=50.65 wf=51.37 r50/100/200/500=56.0/59.0/55.0/51.6 base=51.63 imp=-0.98 cal_gap=4.79 p=0.655561
- REJECTED: gradient_boosting | color_only | val=51.63 hold=50.65 wf=50.72 r50/100/200/500=52.0/54.0/50.5/48.4 base=51.63 imp=-0.98 cal_gap=7.89 p=0.655561
- REJECTED: extra_trees | color_bs_number | val=50.65 hold=50.65 wf=None r50/100/200/500=None/None/None/None base=51.63 imp=-0.98 cal_gap=None p=0.655561
- REJECTED: gradient_boosting | color_number | val=50.65 hold=49.02 wf=None r50/100/200/500=None/None/None/None base=51.63 imp=-2.61 cal_gap=None p=0.83456
- REJECTED: extra_trees | color_transition | val=51.96 hold=48.69 wf=49.02 r50/100/200/500=48.0/54.0/50.5/48.8 base=51.63 imp=-2.94 cal_gap=13.32 p=0.861424
- REJECTED: random_forest | color_transition | val=51.96 hold=48.69 wf=48.89 r50/100/200/500=50.0/55.0/52.5/48.6 base=51.63 imp=-2.94 cal_gap=11.95 p=0.861424
- REJECTED: logistic_regression | previous_color | val=50.0 hold=48.69 wf=None r50/100/200/500=None/None/None/None base=51.63 imp=-2.94 cal_gap=None p=0.861424
- REJECTED: random_forest | previous_color | val=50.0 hold=48.69 wf=None r50/100/200/500=None/None/None/None base=51.63 imp=-2.94 cal_gap=None p=0.861424

## 9. Block stability (best candidate)
```json
[
  {
    "block": 1,
    "n_rounds": 316,
    "color_dist": {
      "RED": 159,
      "GREEN": 157
    },
    "hold_acc": 47.5,
    "baseline": 51.25,
    "improvement": -3.75
  },
  {
    "block": 2,
    "n_rounds": 316,
    "color_dist": {
      "GREEN": 169,
      "RED": 147
    },
    "hold_acc": 53.75,
    "baseline": 53.75,
    "improvement": 0.0
  },
  {
    "block": 3,
    "n_rounds": 316,
    "color_dist": {
      "GREEN": 152,
      "RED": 164
    },
    "hold_acc": 45.0,
    "baseline": 53.75,
    "improvement": -8.75
  },
  {
    "block": 4,
    "n_rounds": 316,
    "color_dist": {
      "RED": 165,
      "GREEN": 151
    },
    "hold_acc": 43.75,
    "baseline": 58.75,
    "improvement": -15.0
  },
  {
    "block": 5,
    "n_rounds": 316,
    "color_dist": {
      "RED": 155,
      "GREEN": 161
    },
    "hold_acc": 45.0,
    "baseline": 51.25,
    "improvement": -6.25
  }
]
```

## 10. Best candidate detail
```json
{
  "MODEL": "logistic_regression",
  "FEATURES": "color_transition",
  "WINDOW": "expanding(all-past)",
  "VALIDATION": 55.23,
  "HOLDOUT": 50.65,
  "WALK_FORWARD": 51.37,
  "RECENT_50": 56.0,
  "RECENT_100": 59.0,
  "RECENT_200": 55.0,
  "RECENT_500": 51.6,
  "BASELINE": 51.63,
  "IMPROVEMENT": -0.98,
  "COVERAGE": 100.0,
  "CALIBRATION": 4.79,
  "STATUS": "REJECTED",
  "pvalue_holdout": 0.655561,
  "selective": [
    {
      "threshold": 0.5,
      "n": 765,
      "coverage": 100.0,
      "accuracy": 51.37,
      "ci95": [
        47.83,
        54.9
      ]
    },
    {
      "threshold": 0.52,
      "n": 161,
      "coverage": 21.05,
      "accuracy": 44.1,
      "ci95": [
        36.66,
        51.82
      ]
    },
    {
      "threshold": 0.55,
      "n": 8,
      "coverage": 1.05,
      "accuracy": 25.0,
      "ci95": [
        7.15,
        59.07
      ]
    },
    {
      "threshold": 0.6,
      "n": 3,
      "coverage": 0.39,
      "accuracy": 33.33,
      "ci95": [
        6.15,
        79.23
      ]
    },
    {
      "threshold": 0.65,
      "n": 0,
      "coverage": 0.0,
      "accuracy": null
    },
    {
      "threshold": 0.7,
      "n": 0,
      "coverage": 0.0,
      "accuracy": null
    }
  ],
  "calibration_detail": {
    "buckets": [
      {
        "bucket": "50-52%",
        "n": 604,
        "avg_prob": 51.25,
        "actual_accuracy": 53.31,
        "gap_pp": -2.06
      },
      {
        "bucket": "52-55%",
        "n": 153,
        "avg_prob": 52.62,
        "actual_accuracy": 45.1,
        "gap_pp": 7.52
      },
      {
        "bucket": "55-60%",
        "n": 5,
        "avg_prob": 56.6,
        "actual_accuracy": 20.0,
        "gap_pp": 36.6
      },
      {
        "bucket": "60-65%",
        "n": 3,
        "avg_prob": 62.37,
        "actual_accuracy": 33.33,
        "gap_pp": 29.04
      },
      {
        "bucket": "65-70%",
        "n": 0
      },
      {
        "bucket": "70%+",
        "n": 0
      }
    ],
    "brier": 0.5019497044605503,
    "log_loss": 0.6951094671383958
  }
}
```

## 11. Verdict notes
- Small improvements (~1–3pp) over majority color can appear by chance.
- Require validation + untouched holdout + walk-forward all supportive for STABLE_CANDIDATE.
- Production must remain unchanged regardless of research outcome.

