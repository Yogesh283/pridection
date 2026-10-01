# Integrity & architecture fixes — final report

## Implementation plan (executed)
A resolution → B WAIT/eval → C naming → D upsert trust → E color canon → F period confirm → G ML honesty → H tests/WF

## Files changed
- `api/color_canon.py` (new)
- `api/client.py`
- `api/history_sync.py`
- `data/database.py`
- `data/migrations.py`
- `data/collector.py`
- `analysis/accuracy.py`
- `models/ensemble.py`
- `models/backtest.py`
- `models/predictor.py`
- `scheduler/runner.py`
- `tools/walkforward_compare.py`
- `tools/repair_resolve_pending.py` (new)
- `tests/test_integrity_fixes.py` (new)
- `exports/INTEGRITY_FIX_REPORT.md`

## Migrations added (additive ALTER only)
- `rounds.source` VARCHAR(32) NULL
- `rounds.color_raw` VARCHAR(64) NULL
- `predictions.decision_strategy` VARCHAR(64) NULL
- `predictions.status` VARCHAR(32) NULL

## P0 fixed
- **F08** Idempotent exact-period resolution; 20 stuck tips resolved; 13 orphans left open (not losses); WAIT unscored
- **F15** Live tip truth: `decision_strategy=majority_w8`; candidates vs final decision explicit; `ml_is_tip_authority=false`

## P1 fixed
- **F01** Trust-ordered upsert (hist_cdn > dearapi > unknown); no duplicate periods
- **F02/F03** Canonical UPPER colors via `canonical_color`
- **F06/F07** Hist-confirmed latest; skip tip if TARGET_NOT_CONFIRMED; never tip already-settled target
- **F14/F22** Future saves use real strategy name (`majority_w8`); historical `ensemble` labels untouched
- **F19/F20** Backtest/WF WAIT = live WAIT; WAIT excluded from accuracy
- **F17/F18** Held-out logged as not OOS; WF label `LIVE_majority_w8`

## P2 fixed
- Explicit `ORDER BY CAST(period AS DECIMAL(30,0))`
- Repair tool dry-run/apply

## Safety answers
| Question | Answer |
|---|---|
| Historical data (game outcomes) changed? | **NO** (repair only resolved pending tips against existing rounds) |
| Historical results rewritten? | **NO** |
| Prediction rows deleted? | **NO** (except duplicate open tips same target — pre-existing) |
| New live strategy | **majority_w8** (WAIT on exact tie) |
| New model_name | Future tips: `majority_w8` (not falsely `ensemble`) |
| Resolution | Exact `target_period`; idempotent; orphans stay open |
| WAIT | Excluded from BS numerator/denominator |
| Period handling | Hist CDN latest required; +1; no tip if unconfirmed/already settled |
| Color | Canonical UPPER; primary RED/GREEN over VIOLET; raw preserved in `color_raw`/`raw_json` |

## Tests
- `test_integrity_fixes`: **8/8 PASS**
- `test_features`: **3/3 PASS**
- `test_api` / `test_backtest`: skipped (pytest not installed in env); integrity covers required cases

## Resolution repair applied
- pending 33 → resolved_now **20**, orphans_left **13**

## Clean walk-forward (majority_w8 live path)
| Metric | Value |
|---|---|
| Big/Small | **49.68%** |
| evaluated (scored) | 618 |
| correct / incorrect | 307 / 311 |
| WAIT | 254 |
| coverage | 70.9% |
| Number | 9.87% |
| Color | 49.19% |
| avg confidence | 0.6495 |
| conf 55–62 | 48.03% (n=381) |
| conf ≥62 | 52.32% (n=237) |

### Baselines (comparison)
| Path | BS | n |
|---|---|---|
| Old last10 live | 47.73% | 639 |
| Majority_w8 + WAIT (this) | **49.68%** | 618 |
| DB settled (after repair, WAIT excluded) | 51.63% | scored 800 / total 802 |

**No claim of material skill beyond coin-flip.** Integrity fixes did not fabricate accuracy. STOP — no further model experiments in this phase.
