# Optimization Architecture (read-only audit)

Generated during Phase 1 of HIGH_ACCURACY optimization.  
**No production behavior changed by this document.**

## Current production tip path

```
scheduler/runner.generate_prediction
  → sync_history_into_db (hist_cdn)
  → resolve_pending_against_rounds
  → EnsemblePredictor.predict(focus=big_small)
  → _pick_big_small → majority_w8 / WAIT on tie
  → save_prediction(model_name=majority_w8) if not WAIT
```

| Layer | Path | Role |
|---|---|---|
| Live tip authority | `models/ensemble.py` `_pick_big_small` | **majority_w8** |
| Candidates (non-authoritative) | majority/frequency/markov/recent/pattern/ml blend | diagnostics / confidence inputs |
| Last-10 analyze | `analysis/big_small_analyze.py` | diagnostic only |
| Features | `models/features.py` | causal past-only when called with `rounds[:i]` |
| Accuracy | `analysis/accuracy.py` | WAIT excluded from BS denom |
| Integrity | `data/database.py`, `api/color_canon.py` | upsert trust, color canon, idempotent resolve |
| Baseline control | majority window=8, tie=WAIT | OOS BS 49.68% n=618 cov 70.9% |

## Modules inventory

- `models/predictor.py` — MajorityWindowModel, Frequency, Markov, Recent, Pattern, MLModelBundle
- `models/ensemble.py` — blend + adaptive weights + final majority override
- `models/backtest.py` — chronological backtest (WAIT skipped)
- `tools/walkforward_compare.py` — multi-method WF
- `scheduler/runner.py` — live loop
- `data/database.py` — rounds/predictions
- `api/history_sync.py` — hist CDN
- `tests/test_integrity_fixes.py` — integrity gate

## Optimization constraints (locked)

1. Integrity fixes remain intact.
2. Chronological walk-forward only; no random split as final evidence.
3. WAIT never scored as win/loss.
4. Production changes only if robust gate passes vs majority_w8.
5. RESEARCH / VALIDATION / HOLDOUT chronological split for candidate selection vs final claim.

## Control baseline (frozen)

| Metric | Value |
|---|---|
| strategy | majority_w8 |
| Big/Small OOS | 49.68% |
| N | 618 |
| WAIT | 254 |
| coverage | 70.9% |
| Number | 9.87% |
| Color | 49.19% |
