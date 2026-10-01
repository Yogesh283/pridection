"""Chronological OOS optimization search vs majority_w8 control.

No future leakage. WAIT excluded from accuracy. Does not change production.
Run: python -m tools.optimize_oos
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label
from api.client import primary_color
from data.database import Database

MIN_HISTORY = 80
# Chronological segments after warm-up index range [MIN_HISTORY, N)
# RESEARCH discovery, VALIDATION selection, HOLDOUT final claim only.


@dataclass
class Result:
    strategy: str
    correct: int
    incorrect: int
    sample_size: int
    wait: int
    accuracy: float
    coverage: float
    avg_confidence: float
    wilson_low: float
    wilson_high: float
    block1: float
    block2: float
    block3: float
    q1: float
    q2: float
    q3: float
    q4: float
    status: str = "EXPERIMENTAL"
    segment: str = "full"


def wilson(ok: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = ok / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = max(0.0, (centre - margin) / denom)
    hi = min(1.0, (centre + margin) / denom)
    return round(100 * lo, 2), round(100 * hi, 2)


def pct(ok: int, n: int) -> float:
    return round(100.0 * ok / n, 2) if n else 0.0


def block_acc(hits: list[int]) -> float:
    if not hits:
        return 0.0
    return round(100.0 * sum(hits) / len(hits), 2)


def evaluate(
    name: str,
    labels: list[str],
    predict_fn: Callable[[int], tuple[str | None, float]],
    *,
    start: int,
    end: int,
    segment: str = "full",
) -> Result:
    """predict_fn(i) uses only labels[:i]; returns (tip|None, confidence)."""
    ok = wait = 0
    hits: list[int] = []
    confs: list[float] = []
    for i in range(start, end):
        tip, conf = predict_fn(i)
        if tip is None:
            wait += 1
            continue
        hit = int(tip == labels[i])
        ok += hit
        hits.append(hit)
        confs.append(conf)
    n = len(hits)
    wrong = n - ok
    total_slots = end - start
    lo, hi = wilson(ok, n)
    # chronological thirds of scored hits
    if n >= 3:
        a, b = n // 3, 2 * n // 3
        b1, b2, b3 = block_acc(hits[:a]), block_acc(hits[a:b]), block_acc(hits[b:])
    else:
        b1 = b2 = b3 = block_acc(hits)
    if n >= 4:
        q = n // 4
        qs = [
            block_acc(hits[0:q]),
            block_acc(hits[q : 2 * q]),
            block_acc(hits[2 * q : 3 * q]),
            block_acc(hits[3 * q :]),
        ]
    else:
        qs = [block_acc(hits)] * 4
    acc = pct(ok, n)
    # stability: max-min across thirds
    spread = max(b1, b2, b3) - min(b1, b2, b3) if n >= 30 else 999.0
    status = "EXPERIMENTAL"
    if n < 50:
        status = "REJECTED"
    elif spread > 12:
        status = "UNSTABLE"
    return Result(
        strategy=name,
        correct=ok,
        incorrect=wrong,
        sample_size=n,
        wait=wait,
        accuracy=acc,
        coverage=pct(n, total_slots),
        avg_confidence=round(sum(confs) / len(confs), 4) if confs else 0.0,
        wilson_low=lo,
        wilson_high=hi,
        block1=b1,
        block2=b2,
        block3=b3,
        q1=qs[0],
        q2=qs[1],
        q3=qs[2],
        q4=qs[3],
        status=status,
        segment=segment,
    )


def majority_at(labels: list[str], i: int, w: int) -> tuple[str | None, float]:
    win = labels[max(0, i - w) : i]
    if not win:
        return None, 0.5
    c = Counter(win)
    b, s = c.get("BIG", 0), c.get("SMALL", 0)
    if b == s:
        return None, 0.5
    tip = "BIG" if b > s else "SMALL"
    conf = (max(b, s) + 1) / (len(win) + 2)
    return tip, conf


def weighted_majority(
    labels: list[str], i: int, w: int, *, mode: str, decay: float
) -> tuple[str | None, float]:
    win = labels[max(0, i - w) : i]
    if not win:
        return None, 0.5
    wb = ws = 0.0
    n = len(win)
    for j, lab in enumerate(win):
        # more recent = higher weight
        age = n - 1 - j
        if mode == "linear":
            wt = (n - age) / n
        else:
            wt = math.exp(-decay * age)
        if lab == "BIG":
            wb += wt
        else:
            ws += wt
    if abs(wb - ws) < 1e-12:
        return None, 0.5
    tip = "BIG" if wb > ws else "SMALL"
    conf = max(wb, ws) / (wb + ws)
    return tip, conf


def follow_at(labels: list[str], i: int) -> tuple[str | None, float]:
    if i < 1:
        return None, 0.5
    return labels[i - 1], 0.5


def flip_at(labels: list[str], i: int) -> tuple[str | None, float]:
    if i < 1:
        return None, 0.5
    return ("SMALL" if labels[i - 1] == "BIG" else "BIG"), 0.5


def frequency_at(labels: list[str], i: int, w: int | None) -> tuple[str | None, float]:
    win = labels[:i] if w is None else labels[max(0, i - w) : i]
    if not win:
        return None, 0.5
    c = Counter(win)
    b, s = c.get("BIG", 0), c.get("SMALL", 0)
    if b == s:
        return None, 0.5
    tip = "BIG" if b > s else "SMALL"
    return tip, (max(b, s) + 1) / (len(win) + 2)


def markov_at(labels: list[str], i: int, order: int) -> tuple[str | None, float]:
    if i < order + 5:
        return None, 0.5
    counts: dict[tuple, Counter] = defaultdict(Counter)
    for t in range(order, i):
        key = tuple(labels[t - order : t])
        counts[key][labels[t]] += 1
    key = tuple(labels[i - order : i])
    c = counts.get(key)
    if not c or sum(c.values()) < 3:
        return None, 0.5
    b, s = c.get("BIG", 0), c.get("SMALL", 0)
    if b == s:
        return None, 0.5
    tip = "BIG" if b > s else "SMALL"
    tot = b + s
    return tip, (max(b, s) + 1) / (tot + 2)


def pattern_at(labels: list[str], i: int, k: int, min_support: int = 5) -> tuple[str | None, float]:
    if i < k + min_support:
        return None, 0.5
    key = tuple(labels[i - k : i])
    nxt = Counter()
    for t in range(k, i):
        if tuple(labels[t - k : t]) == key:
            nxt[labels[t]] += 1
    if sum(nxt.values()) < min_support:
        return None, 0.5
    b, s = nxt.get("BIG", 0), nxt.get("SMALL", 0)
    if b == s:
        return None, 0.5
    tip = "BIG" if b > s else "SMALL"
    tot = b + s
    return tip, (max(b, s) + 1) / (tot + 2)


def streak_len(labels: list[str], i: int) -> int:
    if i < 1:
        return 0
    last = labels[i - 1]
    n = 0
    for x in reversed(labels[:i]):
        if x == last:
            n += 1
        else:
            break
    return n


def streak_flip_at(labels: list[str], i: int, thr: int) -> tuple[str | None, float]:
    if i < 1:
        return None, 0.5
    n = streak_len(labels, i)
    last = labels[i - 1]
    if n >= thr:
        return ("SMALL" if last == "BIG" else "BIG"), min(0.55 + 0.02 * n, 0.7)
    return majority_at(labels, i, 8)


def conditional_prev_at(labels: list[str], i: int, k: int) -> tuple[str | None, float]:
    return pattern_at(labels, i, k, min_support=8)


def adaptive_selector(
    labels: list[str],
    i: int,
    candidates: list[tuple[str, Callable[[int], tuple[str | None, float]]]],
    lookback: int = 40,
) -> tuple[str | None, float]:
    """Pick historically best candidate on [i-lookback, i) then tip for i."""
    start = max(MIN_HISTORY, i - lookback)
    if i - start < 15:
        return majority_at(labels, i, 8)
    best_name = None
    best_acc = -1.0
    for name, fn in candidates:
        ok = tot = 0
        for j in range(start, i):
            tip, _ = fn(j)
            if tip is None:
                continue
            tot += 1
            ok += int(tip == labels[j])
        if tot < 10:
            continue
        acc = ok / tot
        if acc > best_acc:
            best_acc = acc
            best_name = name
    if best_name is None:
        return majority_at(labels, i, 8)
    for name, fn in candidates:
        if name == best_name:
            return fn(i)
    return majority_at(labels, i, 8)


def label_status(r: Result, baseline_acc: float, baseline_n: int) -> str:
    if r.strategy == "majority_w8":
        return "BASELINE"
    if r.sample_size < 80:
        return "REJECTED"
    spread = max(r.block1, r.block2, r.block3) - min(r.block1, r.block2, r.block3)
    if spread > 12:
        return "UNSTABLE"
    # robust beat: higher acc, wilson_low not below baseline wilson-ish, n>=200, cov>=40
    if (
        r.accuracy > baseline_acc + 1.0
        and r.sample_size >= 200
        and r.coverage >= 40.0
        and r.wilson_low >= baseline_acc - 2.0
        and spread <= 10
    ):
        return "ROBUST"
    if r.accuracy > baseline_acc and r.sample_size >= 100 and spread <= 12:
        return "PROMISING"
    if r.accuracy < baseline_acc - 1.0:
        return "REJECTED"
    return "EXPERIMENTAL"


def number_eval(rounds: list[dict], start: int, end: int) -> dict[str, Any]:
    """Exact / top2 / top3 for simple frequency and follow-last-number."""
    out = {}
    for name, mode in [("freq20", 20), ("freq50", 50), ("follow_num", 0)]:
        exact = top2 = top3 = n = 0
        for i in range(start, end):
            hist = [int(r["number"]) for r in rounds[max(0, i - (mode or 1)) : i]]
            if not hist:
                continue
            if mode == 0:
                ranked = [hist[-1]]
            else:
                ranked = [num for num, _ in Counter(hist).most_common(3)]
            actual = int(rounds[i]["number"])
            n += 1
            exact += int(ranked[0] == actual)
            top2 += int(actual in ranked[:2])
            top3 += int(actual in ranked[:3])
        out[name] = {
            "exact": pct(exact, n),
            "top2": pct(top2, n),
            "top3": pct(top3, n),
            "n": n,
        }
    return out


def color_eval(rounds: list[dict], start: int, end: int) -> dict[str, Any]:
    ok = n = 0
    for i in range(start, end):
        hist = [
            primary_color(r.get("color"))
            for r in rounds[max(0, i - 10) : i]
            if primary_color(r.get("color"))
        ]
        if not hist:
            continue
        tip = Counter(hist).most_common(1)[0][0]
        act = primary_color(rounds[i].get("color"))
        if not act:
            continue
        n += 1
        ok += int(tip == act)
    return {"majority_color_w10": {"accuracy": pct(ok, n), "n": n}}


def try_ml_results(labels: list[str], rounds: list[dict], start: int, end: int) -> list[Result]:
    """Lightweight causal ML with expanding train, retrain every K."""
    results: list[Result] = []
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
    except Exception:
        return results

    def make_features(i: int) -> list[float]:
        return [
            1.0 if labels[i - 1] == "BIG" else 0.0,
            1.0 if i >= 2 and labels[i - 2] == "BIG" else 0.0,
            1.0 if i >= 3 and labels[i - 3] == "BIG" else 0.0,
            float(streak_len(labels, i)),
            sum(1 for x in labels[max(0, i - 8) : i] if x == "BIG") / 8.0,
            sum(1 for x in labels[max(0, i - 20) : i] if x == "BIG")
            / max(1, min(20, i)),
        ]

    factories = {
        "ml_logreg": lambda: LogisticRegression(max_iter=500, C=0.5),
        "ml_rf": lambda: RandomForestClassifier(
            n_estimators=60, max_depth=4, min_samples_leaf=10, random_state=42
        ),
        "ml_hgb": lambda: HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=60,
            min_samples_leaf=15,
            random_state=42,
        ),
    }

    for mname, factory in factories.items():
        for retrain_every in (50, 100):
            state: dict[str, Any] = {"clf": None, "last": -10_000}

            def pred_fn(i: int, _factory=factory, _re=retrain_every, _st=state):
                if _st["clf"] is None or i - _st["last"] >= _re:
                    xs, ys = [], []
                    for t in range(MIN_HISTORY, i):
                        xs.append(make_features(t))
                        ys.append(1 if labels[t] == "BIG" else 0)
                    if len(xs) < 60:
                        return majority_at(labels, i, 8)
                    clf = _factory()
                    clf.fit(xs, ys)
                    _st["clf"] = clf
                    _st["last"] = i
                clf = _st["clf"]
                feat = make_features(i)
                classes = list(clf.classes_)
                if 1 not in classes:
                    return majority_at(labels, i, 8)
                p_big = float(clf.predict_proba([feat])[0][classes.index(1)])
                if abs(p_big - 0.5) < 0.02:
                    return None, 0.5
                tip = "BIG" if p_big >= 0.5 else "SMALL"
                return tip, max(p_big, 1.0 - p_big)

            r = evaluate(
                f"{mname}_retrain{retrain_every}",
                labels,
                pred_fn,
                start=start,
                end=end,
                segment="research",
            )
            results.append(r)
    return results


def main() -> int:
    rounds = Database().get_rounds()
    labels = [big_small_label(int(r["number"])) for r in rounds]
    n = len(labels)
    # scored index range
    lo, hi = MIN_HISTORY, n
    span = hi - lo
    # 60% research, 20% validation, 20% holdout
    research_end = lo + int(span * 0.60)
    valid_end = lo + int(span * 0.80)
    holdout_end = hi

    print(
        f"rounds={n} research=[{lo},{research_end}) "
        f"valid=[{research_end},{valid_end}) holdout=[{valid_end},{holdout_end})"
    )

    strategies: list[tuple[str, Callable[[int], tuple[str | None, float]]]] = []

    # A) majority windows
    for w in [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]:
        strategies.append((f"majority_w{w}", lambda i, ww=w: majority_at(labels, i, ww)))

    # B) weighted
    for w in (8, 12, 20):
        strategies.append(
            (f"wlin_w{w}", lambda i, ww=w: weighted_majority(labels, i, ww, mode="linear", decay=0))
        )
        for d in (0.1, 0.2, 0.35):
            strategies.append(
                (
                    f"wexp_w{w}_d{d}",
                    lambda i, ww=w, dd=d: weighted_majority(
                        labels, i, ww, mode="exp", decay=dd
                    ),
                )
            )

    # C) follow / flip
    strategies.append(("follow", lambda i: follow_at(labels, i)))
    strategies.append(("flip", lambda i: flip_at(labels, i)))

    # D) frequency
    for w in (10, 20, 50, 100):
        strategies.append((f"freq_w{w}", lambda i, ww=w: frequency_at(labels, i, ww)))
    strategies.append(("freq_expand", lambda i: frequency_at(labels, i, None)))

    # E) markov
    for order in (1, 2, 3):
        strategies.append((f"markov_o{order}", lambda i, o=order: markov_at(labels, i, o)))

    # F) streak
    for thr in (2, 3, 4, 5):
        strategies.append((f"streak_flip_{thr}", lambda i, t=thr: streak_flip_at(labels, i, t)))

    # G) patterns
    for k in range(2, 9):
        strategies.append((f"pattern_k{k}", lambda i, kk=k: pattern_at(labels, i, kk)))

    # conditional same as pattern with higher support
    for k in (1, 2, 3, 4):
        strategies.append((f"cond_prev{k}", lambda i, kk=k: conditional_prev_at(labels, i, kk)))

    # Research evaluation
    research_rows: list[Result] = []
    for name, fn in strategies:
        r = evaluate(name, labels, fn, start=lo, end=research_end, segment="research")
        research_rows.append(r)
        print(f"research {name:22} acc={r.accuracy:5.2f}% n={r.sample_size:4} cov={r.coverage:5.1f}% wait={r.wait}")

    # ML research (slower)
    print("ML research…")
    ml_rows = try_ml_results(labels, rounds, lo, research_end)
    research_rows.extend(ml_rows)
    for r in ml_rows:
        print(f"research {r.strategy:22} acc={r.accuracy:5.2f}% n={r.sample_size:4}")

    # Adaptive selector using top research candidates (names only — selection is causal inside WF)
    base_cands = [
        ("majority_w8", lambda i: majority_at(labels, i, 8)),
        ("majority_w12", lambda i: majority_at(labels, i, 12)),
        ("flip", lambda i: flip_at(labels, i)),
        ("follow", lambda i: follow_at(labels, i)),
        ("markov_o1", lambda i: markov_at(labels, i, 1)),
        ("freq_w20", lambda i: frequency_at(labels, i, 20)),
    ]
    strategies.append(
        ("adaptive_top", lambda i: adaptive_selector(labels, i, base_cands, 40))
    )
    r_adapt = evaluate(
        "adaptive_top", labels, strategies[-1][1], start=lo, end=research_end, segment="research"
    )
    research_rows.append(r_adapt)
    print(f"research adaptive_top        acc={r_adapt.accuracy:5.2f}% n={r_adapt.sample_size}")

    # Ensemble average of majority_w8 + flip + markov_o1 (causal probs)
    def ensemble_avg(i: int) -> tuple[str | None, float]:
        tips = []
        for fn in (
            lambda j: majority_at(labels, j, 8),
            lambda j: flip_at(labels, j),
            lambda j: markov_at(labels, j, 1),
        ):
            t, c = fn(i)
            if t is not None:
                tips.append((t, c))
        if not tips:
            return None, 0.5
        votes = Counter(t for t, _ in tips)
        if votes.get("BIG", 0) == votes.get("SMALL", 0):
            return None, 0.5
        tip = votes.most_common(1)[0][0]
        conf = sum(c for t, c in tips if t == tip) / len(tips)
        return tip, conf

    r_ens = evaluate(
        "vote_maj8_flip_markov1",
        labels,
        ensemble_avg,
        start=lo,
        end=research_end,
        segment="research",
    )
    research_rows.append(r_ens)

    baseline_r = next(r for r in research_rows if r.strategy == "majority_w8")
    for r in research_rows:
        r.status = label_status(r, baseline_r.accuracy, baseline_r.sample_size)

    # Pick top robust/promising for validation (not using holdout)
    ranked = sorted(
        research_rows,
        key=lambda r: (
            0 if r.status in {"ROBUST", "PROMISING", "BASELINE"} else 1,
            -r.accuracy,
            -r.sample_size,
            -r.wilson_low,
        ),
    )
    # Always include baseline + up to 8 non-rejected
    selected = ["majority_w8"]
    for r in ranked:
        if r.strategy in selected:
            continue
        if r.status == "REJECTED":
            continue
        selected.append(r.strategy)
        if len(selected) >= 9:
            break

    name_to_fn = {name: fn for name, fn in strategies}
    name_to_fn["vote_maj8_flip_markov1"] = ensemble_avg

    print("\nVALIDATION selected:", selected)
    valid_rows: list[Result] = []
    for name in selected:
        fn = name_to_fn.get(name)
        if not fn:
            # rebuild ML names skipped if not in map
            continue
        r = evaluate(name, labels, fn, start=research_end, end=valid_end, segment="validation")
        r.status = label_status(r, baseline_r.accuracy, baseline_r.sample_size)
        valid_rows.append(r)
        print(
            f"valid {name:22} acc={r.accuracy:5.2f}% n={r.sample_size:4} "
            f"ci=[{r.wilson_low},{r.wilson_high}] blocks={r.block1}/{r.block2}/{r.block3}"
        )

    # Choose production candidate from validation: must beat baseline on validation
    base_v = next((r for r in valid_rows if r.strategy == "majority_w8"), None)
    challenger = None
    if base_v:
        for r in sorted(valid_rows, key=lambda x: (-x.accuracy, -x.sample_size)):
            if r.strategy == "majority_w8":
                continue
            if (
                r.accuracy >= base_v.accuracy + 1.0
                and r.sample_size >= 80
                and r.coverage >= 40.0
                and r.status != "UNSTABLE"
                and (max(r.block1, r.block2, r.block3) - min(r.block1, r.block2, r.block3)) <= 12
            ):
                challenger = r
                break

    print("\nHOLDOUT…")
    holdout_names = ["majority_w8"] + ([challenger.strategy] if challenger else [])
    holdout_rows: list[Result] = []
    for name in holdout_names:
        fn = name_to_fn[name]
        r = evaluate(name, labels, fn, start=valid_end, end=holdout_end, segment="holdout")
        holdout_rows.append(r)
        print(
            f"holdout {name:22} acc={r.accuracy:5.2f}% n={r.sample_size:4} "
            f"ci=[{r.wilson_low},{r.wilson_high}]"
        )

    # Full-span baseline confirmation (control)
    full_base = evaluate(
        "majority_w8",
        labels,
        lambda i: majority_at(labels, i, 8),
        start=lo,
        end=hi,
        segment="full",
    )
    print(
        f"\nFULL control majority_w8 acc={full_base.accuracy}% n={full_base.sample_size} "
        f"wait={full_base.wait} cov={full_base.coverage}%"
    )

    # Selective confidence on control
    selective = []
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):

        def sel_fn(i, t=thr):
            tip, conf = majority_at(labels, i, 8)
            if tip is None or conf < t:
                return None, conf
            return tip, conf

        r = evaluate(
            f"majority_w8_conf>={int(thr*100)}",
            labels,
            sel_fn,
            start=lo,
            end=hi,
            segment="full",
        )
        selective.append(r)

    # Number / color
    num_res = number_eval(rounds, lo, hi)
    col_res = color_eval(rounds, lo, hi)

    # Production gate — must beat verified FULL baseline, not a weak segment dip.
    FULL_BASELINE_ACC = full_base.accuracy  # verified control ~49.68
    deploy = False
    final_strategy = "majority_w8"
    final_acc = full_base.accuracy
    gate_reason = "No robust challenger — keep majority_w8"
    if challenger and holdout_rows:
        h_base = next(r for r in holdout_rows if r.strategy == "majority_w8")
        h_ch = next((r for r in holdout_rows if r.strategy == challenger.strategy), None)
        if h_ch is None:
            gate_reason = "Challenger missing on holdout"
        elif h_ch.sample_size < 80:
            gate_reason = f"Holdout N too small ({h_ch.sample_size})"
        elif h_ch.accuracy < FULL_BASELINE_ACC:
            gate_reason = (
                f"Holdout acc {h_ch.accuracy}% < full baseline {FULL_BASELINE_ACC}% "
                "— segment beat alone is not deployable"
            )
        elif h_ch.accuracy < h_base.accuracy + 1.0:
            gate_reason = (
                f"Holdout lift vs segment control too small "
                f"({h_ch.accuracy} vs {h_base.accuracy})"
            )
        elif (
            max(h_ch.block1, h_ch.block2, h_ch.block3)
            - min(h_ch.block1, h_ch.block2, h_ch.block3)
        ) > 12:
            gate_reason = "Holdout block unstable"
        else:
            deploy = True
            final_strategy = challenger.strategy
            final_acc = h_ch.accuracy
            gate_reason = "Challenger beat baseline on validation and holdout vs full control"

    payload = {
        "baseline_full": asdict(full_base),
        "research": [asdict(r) for r in sorted(research_rows, key=lambda x: -x.accuracy)],
        "validation": [asdict(r) for r in valid_rows],
        "holdout": [asdict(r) for r in holdout_rows],
        "selective_majority_w8": [asdict(r) for r in selective],
        "number": num_res,
        "color": col_res,
        "production_gate": {
            "deploy": deploy,
            "challenger": asdict(challenger) if challenger else None,
            "final_strategy": final_strategy,
            "final_oos_accuracy": final_acc,
            "full_baseline_accuracy": FULL_BASELINE_ACC,
            "reason": gate_reason,
        },
        "splits": {
            "min_history": MIN_HISTORY,
            "research": [lo, research_end],
            "validation": [research_end, valid_end],
            "holdout": [valid_end, holdout_end],
        },
        "leakage": "All predict_fn use labels[:i] / rounds[:i] only.",
    }

    out = ROOT / "exports" / "optimize_oos_results.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")
    print("PRODUCTION:", final_strategy, "deploy=", deploy, "acc=", final_acc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
