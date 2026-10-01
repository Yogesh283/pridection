# HIGH ACCURACY OPTIMIZATION REPORT

**Integrity fixes preserved. Production NOT changed.**  
Benchmark: `python -m tools.optimize_oos` → `exports/optimize_oos_results.json`  
Architecture: `exports/OPTIMIZATION_ARCHITECTURE.md`

---

## 1. Current Baseline (CONTROL)

| Field | Value |
|---|---|
| strategy | **majority_w8** (tie = WAIT) |
| Big/Small OOS (full) | **49.68%** |
| N / correct / incorrect | 618 / 307 / 311 |
| WAIT / coverage | 254 / 70.87% |
| Wilson 95% CI | [45.75, 53.61] |
| Blocks (1/2/3) | 50.97 / 52.91 / 45.15 |
| Number | 9.87% (prior integrity WF) |
| Color | 49.19% (prior integrity WF) |

Chronological splits used for research (no peek at holdout for selection):

| Segment | Index range | Role |
|---|---|---|
| RESEARCH | [80, 603) | discovery |
| VALIDATION | [603, 777) | selection |
| HOLDOUT | [777, 952) | final unbiased claim |

---

## 2. Architecture

See `exports/OPTIMIZATION_ARCHITECTURE.md`. Live tip remains `_pick_big_small` → majority_w8. Candidate ensemble/ML are non-authoritative unless production gate passes.

---

## 3. Candidate Strategies Tested

Majority windows 3–50; linear/exp weighted windows; follow/flip; frequency; Markov 1–3; streak-flip; patterns k=2..8; conditional prev; vote ensemble; adaptive selector; ML logreg/RF/HGB retrain 50/100.

Common evaluator returned accuracy, N, WAIT, coverage, Wilson CI, block/quartile stability for every candidate.

---

## 4. Big/Small Results (research leaderboard — raw)

| Strategy | Accuracy | N | Coverage | Correct | Incorrect | WAIT | CI | Blocks | Status |
|----------|----------|---|----------|---------|-----------|------|----|--------|--------|
| adaptive_top | 55.10 | 490 | 93.7 | 270 | 220 | 33 | 50.7–59.5 | 54.6/52.8/57.9 | PROMISING* |
| majority_w4 | 53.13 | 335 | 64.1 | 178 | 157 | 188 | 47.8–58.4 | 49.6/53.6/56.3 | PROMISING* |
| majority_w20 / freq_w20 | 53.07 | 424 | 81.1 | 225 | 199 | 99 | 48.3–57.8 | 55.3/51.8/52.1 | PROMISING* |
| majority_w8 (control on research slice) | 52.79 | 377 | 72.1 | — | — | 146 | 47.7–57.8 | — | BASELINE |
| pattern_k7 | 52.86 | 70 | 13.4 | — | — | 453 | 41.3–64.1 | — | REJECTED (N) |
| ML best (logreg@50) | 51.10 | 409 | — | — | — | — | — | — | EXPERIMENTAL |
| markov_o3 | 46.44 | 463 | 88.5 | — | — | — | — | — | REJECTED |

\*PROMISING on **research only** — must survive validation/holdout.

---

## 5–8. Pattern / Markov / Streak / Regime

- Patterns: low-k ≈48–49%; k=7 looks high only with **N=70** → REJECTED.
- Markov o1–o3: all ≤48.7% on research → no edge.
- Streak-flip: ≤51.8%, not robust vs control.
- Regime/adaptive selector: **55.1% research** then **41.9% validation** → classic overfit / UNSTABLE for production.

---

## 9. ML Results

| Model | Research acc | N |
|---|---|---|
| ml_logreg_retrain50 | 51.10% | 409 |
| ml_logreg_retrain100 | 50.59% | 425 |
| ml_rf_retrain50/100 | ~49.7–50.1% | ~400 |
| ml_hgb_retrain50/100 | ~48.9–49.3% | ~400 |

No ML candidate beat majority_w8 on validation with robust criteria. Held-out-style optimism avoided; chronological OOS used.

---

## 10. Ensemble Results

- `vote_maj8_flip_markov1` evaluated in research run (see JSON).
- Adaptive performance-weighted selector overfit research → failed validation.

---

## 11. Confidence Results (majority_w8 full OOS)

Buckets effectively collapse to tip confidence levels from Laplace majority:

| Threshold | Acc | N | Coverage |
|---|---|---|---|
| ≥55 / ≥60 | 49.68% | 618 | 70.9% |
| ≥65 / ≥70 | 52.32% | 237 | 27.2% |
| ≥75 / ≥80 | 55.56% | 63 | 7.2% |

High-confidence subsets are **small N**; not deployable as primary strategy.

---

## 12. Selective Prediction Results

Same as §11. Selective ≥65 gives +2.6pp with coverage collapse to 27%. Wilson still overlaps 50%. Not a production replacement.

---

## 13. Number Results

| Strategy | Exact | Top-2 | Top-3 | N |
|---|---|---|---|---|
| freq20 | 11.12% | 21.33% | 30.85% | 872 |
| freq50 | 11.47% | 22.36% | 32.34% | 872 |
| follow_num | 8.94% | 8.94% | 8.94% | 872 |

Baseline random exact ≈10%. Slight freq lift is small; **do not confuse top-3 with exact**.

---

## 14. Color Results

| Strategy | Accuracy | N |
|---|---|---|
| majority_color_w10 (primary) | 54.01% | 872 |

Uses existing canonical/primary color path. No definition change.

---

## 15. Stability Analysis

- Full majority_w8 blocks: 51.0 / 52.9 / **45.2** — mild late-block drop; still control.
- adaptive_top: research strong → validation **41.9%** → **UNSTABLE**.
- Validation/holdout era is harder for all strategies (~44–48%); selection on research alone is misleading.

---

## 16. Statistical Validation

- Full control Wilson CI **includes 50%** [45.75, 53.61].
- Research “winners” at 53% have Wilson lows ≈48% — not separable from chance at scale.
- Tiny N patterns (k=7 N=70) rejected despite raw 52.9%.

---

## 17. Validation Results (selected)

| Strategy | Val Acc | N | Notes |
|---|---|---|---|
| wexp_w20_d0.1 | 48.28% | 174 | best of selected on val |
| majority_w8 | 45.38% | 119 | control on same slice |
| adaptive_top | 41.86% | 172 | failed |

Weak era for everyone; slice beat ≠ full-baseline beat.

---

## 18. Final Holdout (untouched)

| Strategy | Holdout Acc | N | CI |
|---|---|---|---|
| majority_w8 | 44.26% | 122 | 35.8–53.1 |
| wexp_w20_d0.1 | 45.14% | 175 | 38.0–52.5 |

Challenger holdout **45.14% < full baseline 49.68%** → **DO NOT DEPLOY**.

---

## 19. Leakage Tests

- Evaluator: `predict_fn(i)` uses only `labels[:i]` / history before i.
- Feature tests (`test_features` no future leakage) remain PASS.
- Integrity suite 8/8 PASS (prior phase).
- No future performance weights in final production path.
- Adaptive selector uses only past lookback (still overfit research dynamics).

---

## 20. Production Decision

| Gate check | Result |
|---|---|
| Genuine full-OOS improvement? | **NO** |
| Sufficient N on holdout with absolute lift vs full baseline? | **NO** |
| Stable across research→validation→holdout? | **NO** (adaptive fails) |
| Leakage? | **None detected** in protocol |
| **Deploy?** | **FALSE** |
| **Production strategy** | **majority_w8** |
| **Genuine OOS accuracy** | **49.68%** (full control WF) |

Status label rules used:

- BASELINE: majority_w8  
- PROMISING: research lift + N≥100 (must re-validate)  
- ROBUST: would require research+validation+holdout beat of full baseline with N/coverage/stability — **none achieved**  
- UNSTABLE: block spread >12 or research/validation collapse  
- REJECTED: N&lt;80 or clear underperformance  

---

## Required Final Answers

1. **Current baseline?** majority_w8, BS OOS **49.68%** (N=618, WAIT=254, cov=70.9%)  
2. **Highest raw OOS (research)?** adaptive_top **55.10%**  
3. **Its N?** 490  
4. **Coverage?** 93.7%  
5. **Wilson CI?** [50.68, 59.45]  
6. **Strongest ROBUST result?** **None** — no ROBUST label earned through validation+holdout vs full baseline  
7. **Did it beat majority_w8 for production?** **NO**  
8. **By how many pp?** N/A for deploy; research adaptive +5.3pp vs research slice control, then **failed** validation (−3.5pp vs val control)  
9. **Block testing survived?** adaptive yes on research; **failed** validation stability/transfer  
10. **Survived validation?** **NO** (adaptive 41.9%; best val challenger still &lt; full baseline)  
11. **Survived untouched holdout vs full baseline?** **NO** (45.14% &lt; 49.68%)  
12. **Leakage detected?** **NO** in the evaluation protocol  
13. **Should production change?** **NO**  
14. **Final production strategy?** **majority_w8**  
15. **Genuine OOS accuracy?** **49.68%** Big/Small  

**STOP.** No further optimization cycle. No production code path change.
