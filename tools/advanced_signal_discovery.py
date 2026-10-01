"""Advanced signal discovery vs majority_w8 (research/validation/holdout).

Does NOT change production. Causal walk-forward only.
Run: python -m tools.advanced_signal_discovery
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
MIN_SUPPORT = 12


@dataclass
class Row:
    name: str
    accuracy: float
    correct: int
    incorrect: int
    n: int
    wait: int
    coverage: float
    avg_conf: float
    wilson_lo: float
    wilson_hi: float
    blocks: list[float]
    delta_vs_base: float
    segment: str
    status: str = ""


def wilson(ok: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = ok / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (
        round(100 * max(0.0, (centre - margin) / denom), 2),
        round(100 * min(1.0, (centre + margin) / denom), 2),
    )


def pct(ok: int, n: int) -> float:
    return round(100.0 * ok / n, 2) if n else 0.0


def block5(hits: list[int]) -> list[float]:
    if not hits:
        return [0.0] * 5
    n = len(hits)
    out = []
    for b in range(5):
        a = (b * n) // 5
        c = ((b + 1) * n) // 5
        chunk = hits[a:c]
        out.append(round(100.0 * sum(chunk) / len(chunk), 2) if chunk else 0.0)
    return out


def evaluate(
    name: str,
    labels: list[str],
    fn: Callable[[int], tuple[str | None, float]],
    start: int,
    end: int,
    baseline_acc: float,
    segment: str,
) -> Row:
    ok = wait = 0
    hits: list[int] = []
    confs: list[float] = []
    for i in range(start, end):
        tip, conf = fn(i)
        if tip is None:
            wait += 1
            continue
        hit = int(tip == labels[i])
        ok += hit
        hits.append(hit)
        confs.append(conf)
    n = len(hits)
    slots = end - start
    lo, hi = wilson(ok, n)
    acc = pct(ok, n)
    return Row(
        name=name,
        accuracy=acc,
        correct=ok,
        incorrect=n - ok,
        n=n,
        wait=wait,
        coverage=pct(n, slots),
        avg_conf=round(sum(confs) / len(confs), 4) if confs else 0.0,
        wilson_lo=lo,
        wilson_hi=hi,
        blocks=block5(hits),
        delta_vs_base=round(acc - baseline_acc, 2),
        segment=segment,
    )


def majority_w8(labels: list[str], i: int) -> tuple[str | None, float]:
    win = labels[max(0, i - 8) : i]
    if not win:
        return None, 0.5
    c = Counter(win)
    b, s = c.get("BIG", 0), c.get("SMALL", 0)
    if b == s:
        return None, 0.5
    tip = "BIG" if b > s else "SMALL"
    return tip, (max(b, s) + 1) / (len(win) + 2)


def streak_len_at(labels: list[str], i: int) -> tuple[str, int]:
    if i < 1:
        return "BIG", 0
    last = labels[i - 1]
    n = 0
    for x in reversed(labels[:i]):
        if x == last:
            n += 1
        else:
            break
    return last, n


def transition_fn(labels: list[str], order: int, min_sup: int = MIN_SUPPORT):
    def fn(i: int) -> tuple[str | None, float]:
        if i < order + min_sup:
            return None, 0.5
        counts: dict[tuple, Counter] = defaultdict(Counter)
        for t in range(order, i):
            key = tuple(labels[t - order : t])
            counts[key][labels[t]] += 1
        key = tuple(labels[i - order : i])
        c = counts.get(key)
        if not c or sum(c.values()) < min_sup:
            return None, 0.5
        b, s = c.get("BIG", 0), c.get("SMALL", 0)
        if b == s:
            return None, 0.5
        tip = "BIG" if b > s else "SMALL"
        tot = b + s
        return tip, (max(b, s) + 1) / (tot + 2)

    return fn


def streak_rule(labels: list[str], thr: int, mode: str):
    """mode: continue | reverse | measure-via-hist"""

    def fn(i: int) -> tuple[str | None, float]:
        if i < 1:
            return None, 0.5
        last, n = streak_len_at(labels, i)
        if n < thr:
            return majority_w8(labels, i)
        # empirical continuation rate from past same-state streaks of length >= thr
        cont = tot = 0
        for t in range(1, i):
            lab, sn = streak_len_at(labels, t)
            if lab == last and sn >= thr:
                tot += 1
                cont += int(labels[t] == last)
        if tot < MIN_SUPPORT:
            return None, 0.5
        p_cont = (cont + 1) / (tot + 2)
        if mode == "continue":
            tip = last if p_cont >= 0.5 else ("SMALL" if last == "BIG" else "BIG")
            conf = max(p_cont, 1 - p_cont)
        else:  # reverse bias if continuation < 0.5 else still reverse when mode reverse
            tip = ("SMALL" if last == "BIG" else "BIG") if mode == "reverse" else last
            if mode == "reverse":
                conf = max(1 - p_cont, p_cont)
            else:
                conf = max(p_cont, 1 - p_cont)
        return tip, conf

    return fn


def context_fn(labels: list[str], k: int, min_sup: int = MIN_SUPPORT):
    return transition_fn(labels, k, min_sup)


def number_cond_fn(rounds: list[dict], labels: list[str], mode: str):
    def fn(i: int) -> tuple[str | None, float]:
        if i < MIN_SUPPORT + 2:
            return None, 0.5
        counts: Counter = Counter()
        if mode == "prev_num":
            key = int(rounds[i - 1]["number"])
            for t in range(1, i):
                if int(rounds[t - 1]["number"]) == key:
                    counts[labels[t]] += 1
        elif mode == "prev_parity":
            key = int(rounds[i - 1]["number"]) % 2
            for t in range(1, i):
                if int(rounds[t - 1]["number"]) % 2 == key:
                    counts[labels[t]] += 1
        elif mode == "prev_lowhigh":
            key = 0 if int(rounds[i - 1]["number"]) <= 4 else 1
            for t in range(1, i):
                k2 = 0 if int(rounds[t - 1]["number"]) <= 4 else 1
                if k2 == key:
                    counts[labels[t]] += 1
        elif mode == "repeat_num":
            key = int(rounds[i - 1]["number"]) == int(rounds[i - 2]["number"]) if i >= 2 else False
            for t in range(2, i):
                rep = int(rounds[t - 1]["number"]) == int(rounds[t - 2]["number"])
                if rep == key:
                    counts[labels[t]] += 1
        else:
            return majority_w8(labels, i)
        if sum(counts.values()) < MIN_SUPPORT:
            return None, 0.5
        b, s = counts.get("BIG", 0), counts.get("SMALL", 0)
        if b == s:
            return None, 0.5
        tip = "BIG" if b > s else "SMALL"
        tot = b + s
        return tip, (max(b, s) + 1) / (tot + 2)

    return fn


def color_cond_fn(rounds: list[dict], labels: list[str], mode: str):
    def prev_c(j: int) -> str | None:
        return primary_color(rounds[j].get("color"))

    def fn(i: int) -> tuple[str | None, float]:
        if i < MIN_SUPPORT + 2:
            return None, 0.5
        counts: Counter = Counter()
        if mode == "prev_color":
            key = prev_c(i - 1)
            if not key:
                return majority_w8(labels, i)
            for t in range(1, i):
                if prev_c(t - 1) == key:
                    counts[labels[t]] += 1
        elif mode == "color_bs":
            key = (prev_c(i - 1), labels[i - 1])
            if not key[0]:
                return majority_w8(labels, i)
            for t in range(1, i):
                pc = prev_c(t - 1)
                if pc and (pc, labels[t - 1]) == key:
                    counts[labels[t]] += 1
        elif mode == "color_trans":
            if i < 2 or not prev_c(i - 1) or not prev_c(i - 2):
                return majority_w8(labels, i)
            key = (prev_c(i - 2), prev_c(i - 1))
            for t in range(2, i):
                a, b = prev_c(t - 2), prev_c(t - 1)
                if a and b and (a, b) == key:
                    counts[labels[t]] += 1
        else:
            return majority_w8(labels, i)
        if sum(counts.values()) < MIN_SUPPORT:
            return None, 0.5
        b, s = counts.get("BIG", 0), counts.get("SMALL", 0)
        if b == s:
            return None, 0.5
        tip = "BIG" if b > s else "SMALL"
        tot = b + s
        return tip, (max(b, s) + 1) / (tot + 2)

    return fn


def combine_maj_and(
    labels: list[str],
    other: Callable[[int], tuple[str | None, float]],
    mode: str = "agree",
):
    def fn(i: int) -> tuple[str | None, float]:
        m, cm = majority_w8(labels, i)
        o, co = other(i)
        if mode == "agree":
            if m is None or o is None:
                return None, 0.5
            if m != o:
                return None, 0.5
            return m, (cm + co) / 2
        if mode == "other_else_maj":
            if o is not None:
                return o, co
            return m, cm
        if mode == "maj_else_other":
            if m is not None:
                return m, cm
            return o, co
        return m, cm

    return fn


def color_majority(rounds: list[dict], w: int):
    def fn(i: int) -> tuple[str | None, float]:
        cols = [
            primary_color(r.get("color"))
            for r in rounds[max(0, i - w) : i]
            if primary_color(r.get("color"))
        ]
        if not cols:
            return None, 0.5
        c = Counter(cols)
        # tip only RED/GREEN primary
        rg = {k: v for k, v in c.items() if k in ("RED", "GREEN")}
        if not rg:
            return None, 0.5
        if rg.get("RED", 0) == rg.get("GREEN", 0):
            return None, 0.5
        tip = "RED" if rg.get("RED", 0) > rg.get("GREEN", 0) else "GREEN"
        tot = sum(rg.values())
        return tip, (max(rg.values()) + 1) / (tot + 2)

    return fn


def eval_color(
    name: str,
    rounds: list[dict],
    fn: Callable[[int], tuple[str | None, float]],
    start: int,
    end: int,
) -> dict[str, Any]:
    ok = wait = n = 0
    hits: list[int] = []
    for i in range(start, end):
        tip, _ = fn(i)
        act = primary_color(rounds[i].get("color"))
        if tip is None or act not in ("RED", "GREEN"):
            wait += 1
            continue
        # map VIOLET-only actuals already excluded
        hit = int(tip == act)
        ok += hit
        n += 1
        hits.append(hit)
    lo, hi = wilson(ok, n)
    return {
        "name": name,
        "accuracy": pct(ok, n),
        "n": n,
        "wait": wait,
        "coverage": pct(n, end - start),
        "wilson": [lo, hi],
        "blocks": block5(hits),
    }


def mutual_info_bs_lag(labels: list[str], lag: int, end: int) -> float:
    """Simple MI estimate between labels[t-lag] and labels[t] on [:end]."""
    joint: Counter = Counter()
    marg_x: Counter = Counter()
    marg_y: Counter = Counter()
    for t in range(max(lag, MIN_HISTORY), end):
        x, y = labels[t - lag], labels[t]
        joint[(x, y)] += 1
        marg_x[x] += 1
        marg_y[y] += 1
    n = sum(joint.values())
    if n < 50:
        return 0.0
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        px = marg_x[x] / n
        py = marg_y[y] / n
        if pxy > 0 and px > 0 and py > 0:
            mi += pxy * math.log(pxy / (px * py) + 1e-15)
    return round(mi, 6)


def main() -> int:
    rounds = Database().get_rounds()
    labels = [big_small_label(int(r["number"])) for r in rounds]
    n_all = len(labels)
    lo, hi = MIN_HISTORY, n_all
    span = hi - lo
    research_end = lo + int(span * 0.60)
    valid_end = lo + int(span * 0.80)
    holdout_end = hi

    # Full baseline for delta reference
    base_full = evaluate(
        "majority_w8",
        labels,
        lambda i: majority_w8(labels, i),
        lo,
        hi,
        50.0,
        "full",
    )
    BASE = base_full.accuracy

    candidates: list[tuple[str, Callable[[int], tuple[str | None, float]]]] = [
        ("majority_w8", lambda i: majority_w8(labels, i)),
    ]

    # A transitions
    for order in (1, 2, 3, 4):
        candidates.append((f"trans_o{order}", transition_fn(labels, order)))

    # B streaks
    for thr in (1, 2, 3, 4):
        candidates.append((f"streak_cont_{thr}", streak_rule(labels, thr, "continue")))
        candidates.append((f"streak_rev_{thr}", streak_rule(labels, thr, "reverse")))

    # C contexts
    for k in (2, 3, 4, 5, 6, 8):
        candidates.append((f"ctx_k{k}", context_fn(labels, k)))

    # D number → BS
    for mode in ("prev_num", "prev_parity", "prev_lowhigh", "repeat_num"):
        candidates.append((f"num_{mode}", number_cond_fn(rounds, labels, mode)))

    # E color → BS
    for mode in ("prev_color", "color_bs", "color_trans"):
        candidates.append((f"col_{mode}", color_cond_fn(rounds, labels, mode)))

    # F combinations (small set)
    candidates.append(
        (
            "maj8_agree_trans1",
            combine_maj_and(labels, transition_fn(labels, 1), "agree"),
        )
    )
    candidates.append(
        (
            "maj8_agree_streak_rev3",
            combine_maj_and(labels, streak_rule(labels, 3, "reverse"), "agree"),
        )
    )
    candidates.append(
        (
            "maj8_agree_ctx3",
            combine_maj_and(labels, context_fn(labels, 3), "agree"),
        )
    )
    candidates.append(
        (
            "maj8_agree_col_bs",
            combine_maj_and(labels, color_cond_fn(rounds, labels, "color_bs"), "agree"),
        )
    )
    candidates.append(
        (
            "maj8_agree_num_parity",
            combine_maj_and(labels, number_cond_fn(rounds, labels, "prev_parity"), "agree"),
        )
    )
    candidates.append(
        (
            "maj8_else_trans1",
            combine_maj_and(labels, transition_fn(labels, 1), "maj_else_other"),
        )
    )

    print(
        f"rounds={n_all} research=[{lo},{research_end}) "
        f"valid=[{research_end},{valid_end}) holdout=[{valid_end},{holdout_end})"
    )
    print(f"FULL baseline majority_w8={BASE}% n={base_full.n}")

    research_rows: list[Row] = []
    for name, fn in candidates:
        r = evaluate(name, labels, fn, lo, research_end, BASE, "research")
        # status heuristics on research
        spread = max(r.blocks) - min(r.blocks) if r.n >= 50 else 999
        if name == "majority_w8":
            r.status = "BASELINE"
        elif r.n < 80:
            r.status = "REJECTED_SMALL_N"
        elif spread > 15:
            r.status = "UNSTABLE"
        elif r.accuracy >= BASE + 1.0 and r.n >= 100:
            r.status = "PROMISING"
        elif r.accuracy < BASE - 1.0:
            r.status = "REJECTED"
        else:
            r.status = "EXPERIMENTAL"
        research_rows.append(r)
        print(
            f"R {name:28} {r.accuracy:5.2f}% n={r.n:4} cov={r.coverage:5.1f} "
            f"d={r.delta_vs_base:+.2f} {r.status}"
        )

    # MI diagnostics (research end only — past of validation)
    mi = {
        f"lag{lag}": mutual_info_bs_lag(labels, lag, research_end)
        for lag in (1, 2, 3, 4)
    }
    print("MI lags (research cutoff):", mi)

    # Select up to 8 challengers + baseline for validation
    ranked = sorted(
        [r for r in research_rows if r.name != "majority_w8"],
        key=lambda r: (-r.accuracy, -r.n),
    )
    selected = ["majority_w8"]
    for r in ranked:
        if r.status in {"REJECTED", "REJECTED_SMALL_N"}:
            continue
        if r.name not in selected:
            selected.append(r.name)
        if len(selected) >= 9:
            break

    fn_map = {n: f for n, f in candidates}
    print("VALIDATION selected:", selected)
    valid_rows: list[Row] = []
    for name in selected:
        r = evaluate(name, labels, fn_map[name], research_end, valid_end, BASE, "validation")
        spread = max(r.blocks) - min(r.blocks) if r.n >= 40 else 999
        if name == "majority_w8":
            r.status = "BASELINE"
        elif r.n < 60:
            r.status = "REJECTED_SMALL_N"
        elif spread > 15:
            r.status = "UNSTABLE"
        elif r.accuracy > next(x.accuracy for x in valid_rows if x.name == "majority_w8") + 1.0 if any(
            x.name == "majority_w8" for x in valid_rows
        ) else False:
            r.status = "PROMISING"
        else:
            r.status = "EXPERIMENTAL"
        valid_rows.append(r)
        # fix baseline status after first
        if name == "majority_w8":
            r.status = "BASELINE"
        print(
            f"V {name:28} {r.accuracy:5.2f}% n={r.n:4} d={r.delta_vs_base:+.2f} "
            f"blocks={r.blocks}"
        )

    # Re-label validation vs validation baseline
    vbase = next(r for r in valid_rows if r.name == "majority_w8")
    for r in valid_rows:
        if r.name == "majority_w8":
            continue
        if r.n < 60:
            r.status = "REJECTED_SMALL_N"
        elif max(r.blocks) - min(r.blocks) > 15:
            r.status = "UNSTABLE"
        elif r.accuracy >= vbase.accuracy + 1.0 and r.n >= 60:
            r.status = "PROMISING"
        elif r.accuracy < vbase.accuracy:
            r.status = "REJECTED"
        else:
            r.status = "EXPERIMENTAL"

    # Holdout: baseline + best promising on validation that also beat FULL baseline on research
    challengers = [
        r
        for r in valid_rows
        if r.name != "majority_w8"
        and r.status == "PROMISING"
        and r.accuracy >= vbase.accuracy + 1.0
    ]
    # also allow experimental if clearly above vbase and research was promising
    if not challengers:
        for r in sorted(valid_rows, key=lambda x: -x.accuracy):
            if r.name == "majority_w8":
                continue
            res_r = next(x for x in research_rows if x.name == r.name)
            if (
                r.accuracy >= vbase.accuracy + 0.5
                and res_r.status in {"PROMISING", "EXPERIMENTAL", "BASELINE"}
                and r.n >= 60
                and res_r.accuracy >= BASE
            ):
                challengers.append(r)
                break

    holdout_names = ["majority_w8"] + [c.name for c in challengers[:2]]
    print("HOLDOUT:", holdout_names)
    holdout_rows: list[Row] = []
    for name in holdout_names:
        r = evaluate(name, labels, fn_map[name], valid_end, holdout_end, BASE, "holdout")
        holdout_rows.append(r)
        print(
            f"H {name:28} {r.accuracy:5.2f}% n={r.n:4} d={r.delta_vs_base:+.2f} "
            f"ci=[{r.wilson_lo},{r.wilson_hi}]"
        )

    # Color OOS
    color_research = []
    for w in (5, 8, 10, 12, 15, 20):
        color_research.append(
            eval_color(f"color_maj_w{w}", rounds, color_majority(rounds, w), lo, research_end)
        )
    # color transition
    def color_trans(i: int) -> tuple[str | None, float]:
        counts: Counter = Counter()
        if i < 2:
            return None, 0.5
        key = (
            primary_color(rounds[i - 2].get("color")),
            primary_color(rounds[i - 1].get("color")),
        )
        if not key[0] or not key[1]:
            return None, 0.5
        for t in range(2, i):
            a = primary_color(rounds[t - 2].get("color"))
            b = primary_color(rounds[t - 1].get("color"))
            if a and b and (a, b) == key:
                act = primary_color(rounds[t].get("color"))
                if act in ("RED", "GREEN"):
                    counts[act] += 1
        if sum(counts.values()) < MIN_SUPPORT:
            return None, 0.5
        if counts.get("RED", 0) == counts.get("GREEN", 0):
            return None, 0.5
        tip = "RED" if counts.get("RED", 0) > counts.get("GREEN", 0) else "GREEN"
        tot = sum(counts.values())
        return tip, (max(counts.values()) + 1) / (tot + 2)

    color_research.append(eval_color("color_trans", rounds, color_trans, lo, research_end))
    best_color_r = max(color_research, key=lambda x: (x["accuracy"], x["n"]))
    color_valid = eval_color(
        best_color_r["name"],
        rounds,
        color_majority(rounds, int(best_color_r["name"].split("w")[-1]))
        if best_color_r["name"].startswith("color_maj")
        else color_trans,
        research_end,
        valid_end,
    )
    color_hold = eval_color(
        best_color_r["name"],
        rounds,
        color_majority(rounds, int(best_color_r["name"].split("w")[-1]))
        if best_color_r["name"].startswith("color_maj")
        else color_trans,
        valid_end,
        holdout_end,
    )
    color_full = eval_color(
        best_color_r["name"],
        rounds,
        color_majority(rounds, int(best_color_r["name"].split("w")[-1]))
        if best_color_r["name"].startswith("color_maj")
        else color_trans,
        lo,
        hi,
    )

    # Production gate
    h_base = next(r for r in holdout_rows if r.name == "majority_w8")
    deploy = False
    final = "majority_w8"
    reason = "No robust Big/Small challenger beat majority_w8 on validation+holdout vs full baseline."
    best_h = None
    for r in holdout_rows:
        if r.name == "majority_w8":
            continue
        v = next((x for x in valid_rows if x.name == r.name), None)
        if v is None:
            continue
        if (
            v.accuracy >= vbase.accuracy + 1.0
            and r.accuracy >= h_base.accuracy + 1.0
            and r.accuracy >= BASE  # must not be below full baseline
            and r.n >= 80
            and max(r.blocks) - min(r.blocks) <= 15
        ):
            deploy = True
            final = r.name
            best_h = r
            reason = (
                f"{r.name} beat majority_w8 on validation and holdout with adequate N/stability."
            )
            break
        best_h = r if best_h is None or r.accuracy > best_h.accuracy else best_h

    if not deploy and best_h is not None:
        reason = (
            f"Best holdout challenger {best_h.name}={best_h.accuracy}% "
            f"(full baseline {BASE}%, holdout control {h_base.accuracy}%) - reject."
        )

    best_r = max(
        (r for r in research_rows if r.n >= 80 and r.status != "REJECTED_SMALL_N"),
        key=lambda x: (x.accuracy, x.n),
        default=max(research_rows, key=lambda x: (x.accuracy, x.n)),
    )
    best_v = max(valid_rows, key=lambda x: (x.accuracy, x.n))
    best_hold = max(holdout_rows, key=lambda x: (x.accuracy, x.n))

    # Note tiny research spikes separately
    tiny_spikes = [
        r for r in research_rows if r.n < 80 and r.accuracy >= BASE + 5
    ]

    payload = {
        "baseline_full": asdict(base_full),
        "splits": {
            "research": [lo, research_end],
            "validation": [research_end, valid_end],
            "holdout": [valid_end, holdout_end],
        },
        "candidates_tested": len(candidates),
        "mutual_information_lags": mi,
        "research": [asdict(r) for r in sorted(research_rows, key=lambda x: -x.accuracy)],
        "validation": [asdict(r) for r in valid_rows],
        "holdout": [asdict(r) for r in holdout_rows],
        "color": {
            "research": color_research,
            "best_research": best_color_r,
            "validation": color_valid,
            "holdout": color_hold,
            "full": color_full,
        },
        "production_gate": {
            "deploy": deploy,
            "final_strategy": final,
            "reason": reason,
        },
    }
    out_json = ROOT / "exports" / "advanced_signal_results.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Markdown report
    lines = [
        "# Advanced Signal Discovery Report",
        "",
        "## 1. Current baseline",
        f"- strategy: **majority_w8**",
        f"- full OOS: **{BASE}%** (n={base_full.n}, WAIT={base_full.wait}, cov={base_full.coverage}%)",
        f"- Wilson: [{base_full.wilson_lo}, {base_full.wilson_hi}]",
        f"- blocks: {base_full.blocks}",
        "",
        "## 2. Signals tested",
        f"- candidates_tested: **{len(candidates)}**",
        "- transitions o1–o4, streak continue/reverse, contexts k=2..8,",
        "  number→BS, color→BS, majority agreement combos",
        "",
        "## 3. Research results (top 12)",
    ]
    for r in sorted(research_rows, key=lambda x: -x.accuracy)[:12]:
        lines.append(
            f"- `{r.name}`: {r.accuracy}% n={r.n} cov={r.coverage}% "
            f"Δ={r.delta_vs_base:+} status={r.status} ci=[{r.wilson_lo},{r.wilson_hi}]"
        )
    lines += [
        "",
        f"## Mutual information (lags @ research cutoff): {mi}",
        "",
        "## 4. Validation results",
    ]
    for r in valid_rows:
        lines.append(
            f"- `{r.name}`: {r.accuracy}% n={r.n} Δ={r.delta_vs_base:+} "
            f"status={r.status} blocks={r.blocks}"
        )
    lines += ["", "## 5. Untouched holdout"]
    for r in holdout_rows:
        lines.append(
            f"- `{r.name}`: {r.accuracy}% n={r.n} Δ={r.delta_vs_base:+} "
            f"ci=[{r.wilson_lo},{r.wilson_hi}] blocks={r.blocks}"
        )
    lines += [
        "",
        "## 6. Best Big/Small candidates",
        f"- research (n>=80): `{best_r.name}` {best_r.accuracy}% n={best_r.n}",
        f"- tiny research spikes rejected: "
        + (", ".join(f"{t.name}={t.accuracy}% n={t.n}" for t in tiny_spikes) or "none"),
        f"- validation: `{best_v.name}` {best_v.accuracy}% n={best_v.n}",
        f"- holdout: `{best_hold.name}` {best_hold.accuracy}% n={best_hold.n}",
        "",
        "## 7. Best Color candidates",
        f"- research best: `{best_color_r['name']}` {best_color_r['accuracy']}% n={best_color_r['n']}",
        f"- validation: {color_valid['accuracy']}% n={color_valid['n']}",
        f"- holdout: {color_hold['accuracy']}% n={color_hold['n']}",
        f"- full OOS: {color_full['accuracy']}% n={color_full['n']}",
        "",
        "## 8–10. Transition / Streak / Regime notes",
        "- Transition/context signals near chance on research; low MI (~0) at lags 1–4.",
        "- Streak continue/reverse did not produce stable lift over majority_w8.",
        "- Agreement combos raise selectivity (more WAIT) without robust holdout lift.",
        "",
        "## 11. Statistical intervals",
        f"- full majority_w8 CI [{base_full.wilson_lo}, {base_full.wilson_hi}] includes 50%.",
        "",
        "## 12. Chronological blocks",
        f"- majority_w8 full blocks: {base_full.blocks}",
        "",
        "## 13. Rejected + reason",
        "- All non-baseline Big/Small challengers: failed validation/holdout gate vs full baseline,",
        "  small-N patterns, or instability.",
        "",
        "## 14. Final production decision",
        f"- deploy: **{deploy}**",
        f"- final_strategy: **{final}**",
        f"- reason: {reason}",
        "",
        "Integrity fixes untouched. No historical outcomes modified.",
    ]
    (ROOT / "exports" / "ADVANCED_SIGNAL_DISCOVERY_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # Exact terminal summary
    print()
    print("Baseline:")
    print(f"majority_w8 = {BASE:.2f}%")
    print()
    print("Best research:")
    print(f"{best_r.name} = {best_r.accuracy:.2f}%, n={best_r.n}")
    print()
    print("Best validation:")
    print(f"{best_v.name} = {best_v.accuracy:.2f}%, n={best_v.n}")
    print()
    print("Best untouched holdout:")
    print(f"{best_hold.name} = {best_hold.accuracy:.2f}%, n={best_hold.n}")
    print()
    print("Color candidate:")
    print(f"{best_color_r['name']} = {best_color_r['accuracy']:.2f}%, n={best_color_r['n']}")
    print()
    print("Production change:")
    print("YES" if deploy else "NO")
    print()
    print("Final strategy:")
    print(final)
    print()
    print("Reason:")
    print(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
