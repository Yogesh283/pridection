"""READ-ONLY deep research: is the COLOR signal real, stable, useful — or noise?

Does NOT change production, thresholds, predictions, round-lock, or schema.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
)

from analysis.statistics import big_small_label
from api.color_canon import canonical_color, primary_from_canonical
from config import COLOR_MAP, EXPORTS_DIR
from data.database import Database

warnings.filterwarnings("ignore")

MIN_HISTORY = 50
COLOR_LABELS = ("RED", "GREEN", "VIOLET")
WF_REFIT_EVERY = 50  # expanding-window walk-forward refit cadence
WINDOW_REFIT_EVERY = 60
TOP_WF = 4
WINDOW_SIZES = (50, 100, 200, 300, 500, 1000, 10_000_000)


def _pct(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return round(100.0 * float(n) / float(d), 2)


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100 * (centre - margin) / denom, 2), round(
        100 * (centre + margin) / denom, 2
    )


def _binom_pvalue_greater(successes: int, n: int, p0: float) -> float | None:
    """One-sided binomial test P(X>=k | Bin(n,p0)) via normal approx with continuity."""
    if n <= 0:
        return None
    mu = n * p0
    var = n * p0 * (1 - p0)
    if var <= 0:
        return None
    z = (successes - 0.5 - mu) / math.sqrt(var)
    # 1 - Phi(z)
    return round(0.5 * math.erfc(z / math.sqrt(2)), 6)


def color_code(c: str | None) -> int:
    return {"RED": 0, "GREEN": 1, "VIOLET": 2}.get(str(c or ""), -1)


def code_to_color(i: int) -> str:
    return COLOR_LABELS[i] if 0 <= i < 3 else "UNK"


def load_rounds(db: Database) -> dict[str, Any]:
    raw = db.get_rounds()
    by_period: dict[str, dict[str, Any]] = {}
    dup = 0
    missing_number = 0
    missing_color = 0
    for r in raw:
        period = str(r.get("period") or "").strip()
        if not period:
            continue
        if period in by_period:
            dup += 1
            if int(r.get("id") or 0) > int(by_period[period].get("id") or 0):
                by_period[period] = r
            continue
        by_period[period] = r

    ordered = sorted(
        by_period.values(),
        key=lambda x: (
            str(x.get("period")),
            str(x.get("created_at") or ""),
            int(x.get("id") or 0),
        ),
    )
    cleaned: list[dict[str, Any]] = []
    for r in ordered:
        try:
            number = int(r["number"])
        except Exception:
            missing_number += 1
            continue
        if number < 0 or number > 9:
            missing_number += 1
            continue
        raw_c = r.get("color")
        canon = canonical_color(raw_c, number)
        primary = primary_from_canonical(canon)
        if not primary:
            missing_color += 1
            canon = canonical_color(None, number)
            primary = primary_from_canonical(canon)
        cleaned.append(
            {
                "id": r.get("id"),
                "period": str(r["period"]),
                "number": number,
                "color_raw": raw_c,
                "color": canon,
                "primary_color": primary,
                "big_small": big_small_label(number),
                "created_at": r.get("created_at"),
            }
        )
    return {
        "total_rounds": len(raw),
        "usable_rounds": len(cleaned),
        "duplicates": dup,
        "missing_numbers": missing_number,
        "missing_colors": missing_color,
        "missing_big_small": 0,
        "range": {
            "min": cleaned[0].get("created_at") if cleaned else None,
            "max": cleaned[-1].get("created_at") if cleaned else None,
            "period_min": cleaned[0]["period"] if cleaned else None,
            "period_max": cleaned[-1]["period"] if cleaned else None,
        },
        "rounds": cleaned,
    }


def verify_mapping(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    mapping_rows = []
    for n in range(10):
        cmap = COLOR_MAP.get(n)
        canon = canonical_color(None, n)
        primary = primary_from_canonical(canon)
        mapping_rows.append(
            {
                "number": n,
                "COLOR_MAP": cmap,
                "canonical": canon,
                "primary": primary,
                "big_small": big_small_label(n),
            }
        )
    mismatches = 0
    checked = 0
    for r in rounds:
        expected = primary_from_canonical(canonical_color(None, int(r["number"])))
        checked += 1
        if r["primary_color"] != expected and not r.get("color_raw"):
            # only count when raw missing (mapped); if raw present, API can differ
            pass
        if r.get("color_raw") in (None, ""):
            if r["primary_color"] != expected:
                mismatches += 1
        else:
            # raw present: primary should match canonical(raw, number)
            got = primary_from_canonical(canonical_color(r.get("color_raw"), r["number"]))
            if got != r["primary_color"]:
                mismatches += 1
    return {
        "sources": [
            "config.py::COLOR_MAP",
            "api/color_canon.py::canonical_color / primary_from_canonical",
            "analysis/statistics.py::big_small_label",
        ],
        "rows": mapping_rows,
        "historical_primary_checks": checked,
        "mismatches": mismatches,
    }


def factories() -> dict[str, Callable[[], Any]]:
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=500, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=70,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            random_state=42,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=90,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            random_state=42,
        ),
    }


FEATURE_SETS = (
    "previous_color",
    "color_history",
    "color_transition",
    "color_frequency",
    "color_only",
    "color_bs",
    "color_number",
    "color_bs_number",
)


def build_feats(history: list[dict[str, Any]], mode: str) -> np.ndarray:
    cols = [r["primary_color"] for r in history]
    nums = [int(r["number"]) for r in history]
    bs = [1 if r["big_small"] == "BIG" else 0 for r in history]
    n = len(history)
    f: list[float] = []

    def lag_color(k: int) -> float:
        return float(color_code(cols[-k])) if n >= k else -1.0

    def lag_num(k: int) -> float:
        return float(nums[-k]) if n >= k else -1.0

    def lag_bs(k: int) -> float:
        return float(bs[-k]) if n >= k else -1.0

    def window_freq(w: int) -> list[float]:
        chunk = cols[-w:] if n else []
        if not chunk:
            return [0.0, 0.0, 0.0]
        c = Counter(chunk)
        return [c.get(lab, 0) / len(chunk) for lab in COLOR_LABELS]

    def streak() -> float:
        if not cols:
            return 0.0
        last = cols[-1]
        s = 0
        for x in reversed(cols):
            if x == last:
                s += 1
            else:
                break
        return float(s)

    include_color = mode in FEATURE_SETS
    # Always compute blocks selectively
    if mode in {
        "previous_color",
        "color_history",
        "color_transition",
        "color_frequency",
        "color_only",
        "color_bs",
        "color_number",
        "color_bs_number",
    }:
        if mode == "previous_color":
            f = [lag_color(1)]
        elif mode == "color_history":
            f = [lag_color(i) for i in range(1, 21)]
            for w in (2, 3, 5, 10, 20):
                f.extend(window_freq(w))
            f.append(streak())
        elif mode == "color_transition":
            f = [
                lag_color(1),
                lag_color(2),
                float(color_code(cols[-2]) * 3 + color_code(cols[-1])) if n >= 2 else -1.0,
                streak(),
                float(cols[-1] == cols[-2]) if n >= 2 else 0.0,
            ]
        elif mode == "color_frequency":
            for w in (5, 10, 20, 50, 100):
                freqs = window_freq(w)
                f.extend(freqs)
                f.append(float(np.argmax(freqs)))
                f.append(float(np.argmin(freqs)))
                f.append(float(max(freqs) - min(freqs)))
        elif mode == "color_only":
            f = [lag_color(i) for i in (1, 2, 3, 5, 10)]
            for w in (5, 10, 20, 50):
                f.extend(window_freq(w))
            f.append(streak())
            f.append(
                float(color_code(cols[-2]) * 3 + color_code(cols[-1])) if n >= 2 else -1.0
            )
        elif mode == "color_bs":
            f = [lag_color(i) for i in (1, 2, 3)]
            f += [lag_bs(i) for i in (1, 2, 3)]
            f.append(streak())
            f.append(
                float(color_code(cols[-2]) * 3 + color_code(cols[-1])) if n >= 2 else -1.0
            )
            f.append(float((bs[-1] != bs[-2])) if n >= 2 else 0.0)
            for w in (5, 10, 20):
                f.extend(window_freq(w))
                chunk_bs = bs[-w:]
                f.append(sum(chunk_bs) / len(chunk_bs) if chunk_bs else 0.5)
        elif mode == "color_number":
            f = [lag_color(i) for i in (1, 2, 3)]
            f += [lag_num(i) for i in (1, 2, 3)]
            f.append(float(nums[-1] % 2) if n else -1.0)
            f.append(float(nums[-1] >= 5) if n else -1.0)
            for w in (5, 10, 20):
                f.extend(window_freq(w))
                chunk_n = nums[-w:]
                f.append(float(np.mean(chunk_n)) if chunk_n else -1.0)
        elif mode == "color_bs_number":
            f = [lag_color(i) for i in (1, 2, 3)]
            f += [lag_bs(i) for i in (1, 2, 3)]
            f += [lag_num(i) for i in (1, 2, 3)]
            f.append(streak())
            f.append(float(nums[-1] % 2) if n else -1.0)
            for w in (5, 10, 20):
                f.extend(window_freq(w))
        else:
            f = [lag_color(1)]
    return np.asarray(f, dtype=float)


def build_matrix(
    rounds: list[dict[str, Any]], mode: str, min_history: int = MIN_HISTORY
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    idxs: list[int] = []
    for i in range(min_history, len(rounds)):
        xs.append(build_feats(rounds[:i], mode))
        ys.append(color_code(rounds[i]["primary_color"]))
        idxs.append(i)
    x = np.vstack(xs) if xs else np.empty((0, 1))
    y = np.asarray(ys, dtype=int)
    return x, y, idxs


def chrono_split(n: int) -> tuple[slice, slice, slice]:
    n_tr = int(n * 0.6)
    n_va = int(n * 0.2)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def eval_acc(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {"n": 0, "accuracy": None, "balanced_accuracy": None}
    acc = accuracy_score(y_true, y_pred)
    try:
        bal = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal = acc
    wins = int((y_true == y_pred).sum())
    return {
        "n": int(len(y_true)),
        "accuracy": round(100 * acc, 2),
        "balanced_accuracy": round(100 * bal, 2),
        "wins": wins,
        "ci95": _wilson(wins, len(y_true)),
        "pvalue_vs_majority": None,  # filled by caller
    }


def fit_pred(model: Any, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray):
    model.fit(xtr, ytr)
    pred = model.predict(xte)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(xte)
        except Exception:
            proba = None
    return pred, proba


def majority_baseline(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return 100.0 * Counter(y.tolist()).most_common(1)[0][1] / len(y)


def transition_tables(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    cols = [r["primary_color"] for r in rounds]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for a, b in zip(cols, cols[1:]):
        counts[a][b] += 1
    table = {}
    for a in COLOR_LABELS:
        tot = sum(counts[a].values())
        table[a] = {
            b: {
                "count": int(counts[a][b]),
                "prob": round(counts[a][b] / tot, 4) if tot else None,
            }
            for b in COLOR_LABELS
        }
        table[a]["_n"] = tot
    # lag-1 markov accuracy in-sample descriptive
    hits = sum(1 for a, b in zip(cols, cols[1:]) if counts[a] and b == counts[a].most_common(1)[0][0])
    return {
        "transitions": table,
        "in_sample_markov_acc": _pct(hits, max(0, len(cols) - 1)),
    }


def streak_analysis(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    cols = [r["primary_color"] for r in rounds]
    by_streak: dict[str, Counter[str]] = defaultdict(Counter)
    cont_hits = 0
    rev_hits = 0
    tot = 0
    for i in range(1, len(cols)):
        last = cols[i - 1]
        s = 1
        j = i - 2
        while j >= 0 and cols[j] == last:
            s += 1
            j -= 1
        key = str(min(s, 5)) if s < 5 else ">=5"
        by_streak[key][cols[i]] += 1
        tot += 1
        if cols[i] == last:
            cont_hits += 1
        else:
            rev_hits += 1
    out = {}
    for k, cnt in sorted(by_streak.items(), key=lambda x: x[0]):
        n = sum(cnt.values())
        out[k] = {
            "n": n,
            "dist": {c: {"count": int(v), "pct": _pct(v, n)} for c, v in cnt.items()},
            "continue_rate_if_same_as_prev_overall": None,
        }
    return {
        "by_streak": out,
        "continue_prev_rate": _pct(cont_hits, tot),
        "change_rate": _pct(rev_hits, tot),
        "n": tot,
    }


def walk_forward(
    rounds: list[dict[str, Any]],
    mode: str,
    model_name: str,
    start: int | None = None,
    refit_every: int = WF_REFIT_EVERY,
) -> dict[str, Any]:
    fac = factories()[model_name]
    x_all, y_all, idxs = build_matrix(rounds, mode)
    # idxs[j] is round index; sample j uses history rounds[:idxs[j]]
    n_samples = len(y_all)
    # start predicting from mid series sample index
    start_j = 0
    if start is None:
        start_j = max(80, int(n_samples * 0.5))
    else:
        # map round index to sample index
        start_j = next((j for j, ri in enumerate(idxs) if ri >= start), int(n_samples * 0.5))

    preds: list[int] = []
    actuals: list[int] = []
    probas: list[np.ndarray] = []
    model = None
    last_fit = -10_000
    for j in range(start_j, n_samples):
        if model is None or (j - last_fit) >= refit_every:
            if j < 60:
                continue
            model = fac()
            model.fit(x_all[:j], y_all[:j])
            last_fit = j
        assert model is not None
        row = x_all[j : j + 1]
        pred = int(model.predict(row)[0])
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(row)[0]
            except Exception:
                proba = None
        preds.append(pred)
        actuals.append(int(y_all[j]))
        if proba is not None:
            full = np.zeros(3, dtype=float)
            classes = list(model.classes_)
            for ci, c in enumerate(classes):
                full[int(c)] = float(proba[ci])
            probas.append(full)
        else:
            probas.append(np.array([1 / 3, 1 / 3, 1 / 3]))
    y_true = np.asarray(actuals, dtype=int)
    y_pred = np.asarray(preds, dtype=int)
    metrics = eval_acc(y_true, y_pred)
    recent = {}
    for w in (50, 100, 200, 500):
        if len(y_true) >= max(20, w // 5):
            yy = y_true[-w:] if len(y_true) >= w else y_true
            pp = y_pred[-w:] if len(y_pred) >= w else y_pred
            recent[f"recent_{w}"] = eval_acc(yy, pp).get("accuracy")
        else:
            recent[f"recent_{w}"] = None
    return {
        "metrics": metrics,
        "recent": recent,
        "y_true": y_true,
        "y_pred": y_pred,
        "probas": np.vstack(probas) if probas else np.empty((0, 3)),
        "start_index": start_j,
    }


def selective_thresholds(y_true: np.ndarray, probas: np.ndarray) -> list[dict[str, Any]]:
    if len(y_true) == 0 or len(probas) == 0:
        return []
    conf = probas.max(axis=1)
    pred = probas.argmax(axis=1)
    rows = []
    for thr in (0.50, 0.52, 0.55, 0.60, 0.65, 0.70):
        mask = conf >= thr
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "threshold": thr,
                    "n": 0,
                    "coverage": 0.0,
                    "accuracy": None,
                }
            )
            continue
        acc = accuracy_score(y_true[mask], pred[mask])
        rows.append(
            {
                "threshold": thr,
                "n": n,
                "coverage": _pct(n, len(y_true)),
                "accuracy": round(100 * acc, 2),
                "ci95": _wilson(int((y_true[mask] == pred[mask]).sum()), n),
            }
        )
    return rows


def calibration_buckets(y_true: np.ndarray, probas: np.ndarray) -> list[dict[str, Any]]:
    if len(y_true) == 0 or len(probas) == 0:
        return []
    conf = probas.max(axis=1)
    pred = probas.argmax(axis=1)
    buckets = [
        (0.50, 0.52, "50-52%"),
        (0.52, 0.55, "52-55%"),
        (0.55, 0.60, "55-60%"),
        (0.60, 0.65, "60-65%"),
        (0.65, 0.70, "65-70%"),
        (0.70, 1.01, "70%+"),
    ]
    out = []
    for lo, hi, name in buckets:
        mask = (conf >= lo) & (conf < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"bucket": name, "n": 0})
            continue
        acc = accuracy_score(y_true[mask], pred[mask])
        out.append(
            {
                "bucket": name,
                "n": n,
                "avg_prob": round(float(conf[mask].mean()) * 100, 2),
                "actual_accuracy": round(100 * acc, 2),
                "gap_pp": round(100 * (float(conf[mask].mean()) - acc), 2),
            }
        )
    # overall brier (multiclass one-vs-rest average)
    try:
        y_oh = np.eye(3)[y_true]
        brier = float(np.mean(np.sum((probas - y_oh) ** 2, axis=1)))
        ll = float(log_loss(y_true, probas, labels=[0, 1, 2]))
    except Exception:
        brier, ll = None, None
    return {"buckets": out, "brier": brier, "log_loss": ll}


def block_stability(
    rounds: list[dict[str, Any]], mode: str, model_name: str, n_blocks: int = 5
) -> list[dict[str, Any]]:
    n = len(rounds)
    size = n // n_blocks
    fac = factories()[model_name]
    rows = []
    for b in range(n_blocks):
        start = b * size
        end = n if b == n_blocks - 1 else (b + 1) * size
        chunk = rounds[start:end]
        if len(chunk) < MIN_HISTORY + 40:
            continue
        x, y, _ = build_matrix(chunk, mode)
        if len(y) < 40:
            continue
        split = int(len(y) * 0.7)
        model = fac()
        pred, _ = fit_pred(model, x[:split], y[:split], x[split:])
        m = eval_acc(y[split:], pred)
        base = majority_baseline(y[split:])
        dist = Counter([r["primary_color"] for r in chunk])
        rows.append(
            {
                "block": b + 1,
                "n_rounds": len(chunk),
                "color_dist": dict(dist),
                "hold_acc": m.get("accuracy"),
                "baseline": round(base, 2),
                "improvement": round((m.get("accuracy") or 0) - base, 2),
            }
        )
    return rows


def status_for(
    val_acc: float | None,
    hold_acc: float | None,
    wf_acc: float | None,
    baseline: float,
    recent: dict[str, Any],
    cal_gap_mean: float | None,
) -> str:
    if hold_acc is None or val_acc is None:
        return "REJECTED"
    if hold_acc < baseline:
        return "REJECTED"
    imp_h = hold_acc - baseline
    imp_v = val_acc - baseline
    # collapse checks
    if val_acc >= baseline + 3 and hold_acc < baseline + 0.5:
        return "UNSTABLE"
    if hold_acc >= baseline + 3 and (wf_acc is not None and wf_acc < baseline):
        return "UNSTABLE"
    rec_vals = [v for v in recent.values() if v is not None]
    if rec_vals and max(rec_vals) - min(rec_vals) > 12:
        # only reject as unstable if mean also weak
        if sum(rec_vals) / len(rec_vals) < baseline + 0.5:
            return "UNSTABLE"
    if cal_gap_mean is not None and cal_gap_mean > 8:
        # overconfident
        if imp_h < 2:
            return "WEAK"
    if (
        imp_h >= 2.0
        and imp_v >= 1.0
        and wf_acc is not None
        and wf_acc >= baseline
        and (not rec_vals or sum(1 for v in rec_vals if v >= baseline) >= max(1, len(rec_vals) - 1))
    ):
        return "STABLE_CANDIDATE"
    if imp_h >= 1.0 and wf_acc is not None and wf_acc >= baseline - 0.5:
        return "PROMISING"
    if imp_h >= 0.5:
        return "WEAK"
    return "REJECTED"


def main() -> int:
    print("COLOR deep research (READ-ONLY)...", flush=True)
    db = Database()
    data = load_rounds(db)
    rounds = data["rounds"]
    mapping = verify_mapping(rounds)
    if len(rounds) < MIN_HISTORY + 150:
        print("Insufficient data")
        return 1

    colors = [r["primary_color"] for r in rounds]
    dist = Counter(colors)
    total = len(colors)
    color_dist = {
        c: {"count": int(dist[c]), "pct": _pct(dist[c], total)} for c in COLOR_LABELS
    }
    majority_color = dist.most_common(1)[0][0]
    color_baseline = _pct(dist[majority_color], total) or 0.0
    random_baseline = round(100 / 3, 2)

    print(
        f"Usable={data['usable_rounds']} baseline={color_baseline}% majority={majority_color}",
        flush=True,
    )

    transitions = transition_tables(rounds)
    streaks = streak_analysis(rounds)

    # --- holdout/val for each feature mode x model ---
    championship: list[dict[str, Any]] = []
    ablation: dict[str, Any] = {}

    # Select champion on validation only, then score holdout once + walk-forward
    print("Ablation + model grid (chrono val)...", flush=True)
    for mode in FEATURE_SETS:
        x, y, _ = build_matrix(rounds, mode)
        tr, va, ho = chrono_split(len(y))
        base_ho = majority_baseline(y[ho])
        base_va = majority_baseline(y[va])
        mode_rows = []
        for mname, fac in factories().items():
            try:
                model = fac()
                pred_va, proba_va = fit_pred(model, x[tr], y[tr], x[va])
                model2 = fac()
                pred_ho, proba_ho = fit_pred(model2, x[tr], y[tr], x[ho])
            except Exception as exc:  # noqa: BLE001
                mode_rows.append({"model": mname, "error": str(exc)})
                continue
            val_m = eval_acc(y[va], pred_va)
            ho_m = eval_acc(y[ho], pred_ho)
            mode_rows.append(
                {
                    "model": mname,
                    "validation": val_m,
                    "holdout": ho_m,
                    "baseline_holdout": round(base_ho, 2),
                    "improvement_holdout": round(
                        (ho_m.get("accuracy") or 0) - base_ho, 2
                    ),
                    "pvalue_holdout": _binom_pvalue_greater(
                        ho_m.get("wins") or 0,
                        ho_m.get("n") or 0,
                        base_ho / 100.0,
                    ),
                }
            )
        ablation[mode] = mode_rows

    # Pick best by validation accuracy among those with holdout >= baseline - 0.5
    candidates = []
    for mode, rows in ablation.items():
        for r in rows:
            if "error" in r:
                continue
            candidates.append({**r, "features": mode})
    candidates.sort(
        key=lambda r: float((r.get("validation") or {}).get("accuracy") or 0),
        reverse=True,
    )

    # Top candidates for expensive walk-forward (limit)
    top = candidates[:TOP_WF]
    print(f"Walk-forward on top {len(top)} candidates...", flush=True)
    wf_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for cand in top:
        key = (cand["features"], cand["model"])
        print(f"  WF {key[0]} / {key[1]}", flush=True)
        wf_cache[key] = walk_forward(rounds, cand["features"], cand["model"])

    # Rolling window size study — use fast logistic model only
    best_mode_for_window = top[0]["features"] if top else "color_bs"
    best_model_for_window = "logistic_regression"
    print("Rolling train-window study (logistic)...", flush=True)
    window_rows = []
    x_all, y_all, idxs = build_matrix(rounds, best_mode_for_window)
    test_start = int(len(y_all) * 0.8)
    for w in WINDOW_SIZES:
        preds = []
        acts = []
        fac = factories()[best_model_for_window]
        model = None
        last = -999
        for ti in range(test_start, len(y_all)):
            lo = 0 if w >= 10_000_000 else max(0, ti - w)
            if ti - lo < 40:
                continue
            if model is None or (ti - last) >= WINDOW_REFIT_EVERY:
                model = fac()
                model.fit(x_all[lo:ti], y_all[lo:ti])
                last = ti
            pred = int(model.predict(x_all[ti : ti + 1])[0])
            preds.append(pred)
            acts.append(int(y_all[ti]))
        m = eval_acc(np.asarray(acts), np.asarray(preds))
        window_rows.append(
            {
                "window": "all" if w >= 10_000_000 else w,
                **m,
            }
        )
        print(f"  window={window_rows[-1]['window']} acc={m.get('accuracy')}", flush=True)

    # Build championship table with WF where available
    print("Building championship table...", flush=True)
    for cand in candidates[:8]:
        key = (cand["features"], cand["model"])
        wf = wf_cache.get(key)
        recent = (wf or {}).get("recent") or {}
        hold_acc = (cand.get("holdout") or {}).get("accuracy")
        val_acc = (cand.get("validation") or {}).get("accuracy")
        wf_acc = ((wf or {}).get("metrics") or {}).get("accuracy")
        base = cand.get("baseline_holdout") or color_baseline
        cal = None
        cal_gap = None
        if wf and len(wf.get("probas", [])) > 0:
            cal = calibration_buckets(wf["y_true"], wf["probas"])
            gaps = [
                abs(b["gap_pp"])
                for b in cal.get("buckets", [])
                if b.get("n", 0) >= 10 and "gap_pp" in b
            ]
            cal_gap = float(np.mean(gaps)) if gaps else None
        st = status_for(val_acc, hold_acc, wf_acc, float(base), recent, cal_gap)
        sel = (
            selective_thresholds(wf["y_true"], wf["probas"])
            if wf and len(wf.get("probas", []))
            else []
        )
        championship.append(
            {
                "MODEL": cand["model"],
                "FEATURES": cand["features"],
                "WINDOW": "expanding(all-past)",
                "VALIDATION": val_acc,
                "HOLDOUT": hold_acc,
                "WALK_FORWARD": wf_acc,
                "RECENT_50": recent.get("recent_50"),
                "RECENT_100": recent.get("recent_100"),
                "RECENT_200": recent.get("recent_200"),
                "RECENT_500": recent.get("recent_500"),
                "BASELINE": base,
                "IMPROVEMENT": round((hold_acc or 0) - float(base), 2),
                "COVERAGE": 100.0,
                "CALIBRATION": round(cal_gap, 2) if cal_gap is not None else None,
                "STATUS": st if wf_acc is not None else ("WEAK" if (hold_acc or 0) >= float(base) else "REJECTED"),
                "pvalue_holdout": cand.get("pvalue_holdout"),
                "selective": sel,
                "calibration_detail": cal,
            }
        )

    # Prefer STABLE_CANDIDATE then PROMISING then best holdout improvement
    rank = {
        "STABLE_CANDIDATE": 0,
        "PROMISING": 1,
        "WEAK": 2,
        "UNSTABLE": 3,
        "REJECTED": 4,
    }
    championship.sort(
        key=lambda r: (
            rank.get(r["STATUS"], 9),
            -(r.get("IMPROVEMENT") or -999),
            -(r.get("WALK_FORWARD") or -999),
        )
    )
    best = championship[0] if championship else None

    # Block stability for best
    blocks = []
    if best:
        blocks = block_stability(rounds, best["FEATURES"], best["MODEL"])

    # Signal group flags from ablation (best holdout per group)
    def best_imp(mode: str) -> float:
        rows = ablation.get(mode) or []
        imps = [r.get("improvement_holdout") for r in rows if "error" not in r]
        return max(imps) if imps else -999

    def flag(mode: str) -> str:
        imp = best_imp(mode)
        # also need val not collapsing — approximate via any model with val>=baseline
        rows = [r for r in (ablation.get(mode) or []) if "error" not in r]
        ok = False
        for r in rows:
            va = (r.get("validation") or {}).get("accuracy") or 0
            ho = (r.get("holdout") or {}).get("accuracy") or 0
            if ho >= color_baseline and va >= color_baseline - 1:
                ok = True
        if imp >= 2 and ok:
            return "FOUND"
        if imp >= 0.5 and ok:
            return "WEAK"
        return "NOT_DETECTED"

    # Final signal status
    if best and best["STATUS"] == "STABLE_CANDIDATE":
        signal_status = "REAL"
        banner = "STABLE COLOR CANDIDATE FOUND"
    elif best and best["STATUS"] == "PROMISING":
        signal_status = "WEAK"
        banner = "COLOR SIGNAL NOT STABLE"
    elif best and best["STATUS"] in {"WEAK", "UNSTABLE"}:
        signal_status = "UNSTABLE" if best["STATUS"] == "UNSTABLE" else "WEAK"
        banner = "COLOR SIGNAL NOT STABLE"
    else:
        signal_status = "NOT_DETECTED"
        banner = "COLOR SIGNAL NOT STABLE"

    # Stricter: require val+hold+wf all beat baseline for REAL
    if best:
        v, h, w = best.get("VALIDATION"), best.get("HOLDOUT"), best.get("WALK_FORWARD")
        base = float(best.get("BASELINE") or color_baseline)
        if not (
            v is not None
            and h is not None
            and w is not None
            and v >= base
            and h >= base + 1.0
            and w >= base
        ):
            if signal_status == "REAL":
                signal_status = "WEAK"
                banner = "COLOR SIGNAL NOT STABLE"

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = EXPORTS_DIR / "COLOR_SIGNAL_DEEP_RESEARCH_REPORT.md"
    lines = [
        "# Color Signal Deep Research Report",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "**READ-ONLY. Production UNCHANGED. No model activation.**",
        "",
        f"## Banner: {banner}",
        "",
        "## 1. Color mapping (from application code)",
        f"- Sources: {', '.join(mapping['sources'])}",
        "",
        "| Number | COLOR_MAP | Canonical | Primary | BIG/SMALL |",
        "|---:|---|---|---|---|",
    ]
    for row in mapping["rows"]:
        lines.append(
            f"| {row['number']} | {row['COLOR_MAP']} | {row['canonical']} | {row['primary']} | {row['big_small']} |"
        )
    lines += [
        "",
        f"- Historical primary consistency checks: {mapping['historical_primary_checks']} mismatches={mapping['mismatches']}",
        "",
        "## 2. Dataset audit",
        f"- total_rounds: {data['total_rounds']}",
        f"- usable_rounds: {data['usable_rounds']}",
        f"- duplicates: {data['duplicates']}",
        f"- missing_numbers: {data['missing_numbers']}",
        f"- missing_colors: {data['missing_colors']}",
        f"- missing_big_small: {data['missing_big_small']}",
        f"- range: {data['range']}",
        "",
        "## 3. Color baseline",
        f"- distribution: {json.dumps(color_dist)}",
        f"- majority_color: {majority_color}",
        f"- majority_baseline: {color_baseline}%",
        f"- random_baseline: {random_baseline}%",
        "",
        "## 4. Color transitions",
        f"```json\n{json.dumps(transitions, indent=2)}\n```",
        "",
        "## 5. Streak analysis",
        f"```json\n{json.dumps(streaks, indent=2)}\n```",
        "",
        "## 6. Feature ablation (best holdout improvement per group)",
    ]
    for mode in FEATURE_SETS:
        lines.append(f"- {mode}: best_improvement={best_imp(mode)} flag={flag(mode)}")

    lines += [
        "",
        "## 7. Rolling window study",
        f"- feature/model locked from val selection: {best_mode_for_window} / {best_model_for_window}",
    ]
    for w in window_rows:
        lines.append(
            f"- window={w['window']}: n={w['n']} acc={w['accuracy']}% bal={w['balanced_accuracy']}% ci={w['ci95']}"
        )

    lines += ["", "## 8. Championship table"]
    for row in championship:
        lines.append(
            f"- {row['STATUS']}: {row['MODEL']} | {row['FEATURES']} | "
            f"val={row['VALIDATION']} hold={row['HOLDOUT']} wf={row['WALK_FORWARD']} "
            f"r50/100/200/500={row['RECENT_50']}/{row['RECENT_100']}/{row['RECENT_200']}/{row['RECENT_500']} "
            f"base={row['BASELINE']} imp={row['IMPROVEMENT']} cal_gap={row['CALIBRATION']} p={row['pvalue_holdout']}"
        )

    lines += [
        "",
        "## 9. Block stability (best candidate)",
        f"```json\n{json.dumps(blocks, indent=2)}\n```",
        "",
        "## 10. Best candidate detail",
        f"```json\n{json.dumps(best, indent=2, default=str)[:15000]}\n```",
        "",
        "## 11. Verdict notes",
        "- Small improvements (~1–3pp) over majority color can appear by chance.",
        "- Require validation + untouched holdout + walk-forward all supportive for STABLE_CANDIDATE.",
        "- Production must remain unchanged regardless of research outcome.",
        "",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (EXPORTS_DIR / "COLOR_SIGNAL_DEEP_RESEARCH_SUMMARY.json").write_text(
        json.dumps(
            {
                "banner": banner,
                "signal_status": signal_status,
                "baseline": color_baseline,
                "best": best,
                "dataset": data["usable_rounds"],
                "flags": {
                    "transition": "FOUND"
                    if (transitions.get("in_sample_markov_acc") or 0) >= color_baseline + 1
                    else "NOT_DETECTED",
                    "streak": "FOUND"
                    if abs((streaks.get("continue_prev_rate") or 50) - 50) >= 5
                    else "NOT_DETECTED",
                    "frequency": flag("color_frequency"),
                    "color_bs": flag("color_bs"),
                    "color_number": flag("color_number"),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    # Final exact print block
    b = best or {}
    print("DATASET:")
    print(f"{data['usable_rounds']} usable rounds | supervised~{data['usable_rounds'] - MIN_HISTORY}")
    print("")
    print("COLOR_BASELINE:")
    print(f"{color_baseline}% (majority={majority_color}; random={random_baseline}%)")
    print("")
    print("BEST_COLOR_MODEL:")
    print(b.get("MODEL"))
    print("")
    print("BEST_FEATURES:")
    print(b.get("FEATURES"))
    print("")
    print("BEST_WINDOW:")
    print(b.get("WINDOW"))
    print("")
    print("VALIDATION_ACCURACY:")
    print(f"{b.get('VALIDATION')}%")
    print("")
    print("HOLDOUT_ACCURACY:")
    print(f"{b.get('HOLDOUT')}%")
    print("")
    print("WALK_FORWARD_ACCURACY:")
    print(f"{b.get('WALK_FORWARD')}%")
    print("")
    print("RECENT_50:")
    print(f"{b.get('RECENT_50')}%")
    print("")
    print("RECENT_100:")
    print(f"{b.get('RECENT_100')}%")
    print("")
    print("RECENT_200:")
    print(f"{b.get('RECENT_200')}%")
    print("")
    print("RECENT_500:")
    print(f"{b.get('RECENT_500')}%")
    print("")
    print("IMPROVEMENT_OVER_BASELINE:")
    print(f"{b.get('IMPROVEMENT')} pp")
    print("")
    print("COLOR_TRANSITION_SIGNAL:")
    print(
        "FOUND"
        if (transitions.get("in_sample_markov_acc") or 0) >= color_baseline + 1
        else "NOT_DETECTED"
    )
    print("")
    print("COLOR_STREAK_SIGNAL:")
    # OOS: continue-prev as predictor on holdout portion of series
    print(
        "WEAK"
        if abs((streaks.get("continue_prev_rate") or 50) - (100 / 3)) > 5
        else "NOT_DETECTED"
    )
    print("")
    print("COLOR_FREQUENCY_SIGNAL:")
    print(flag("color_frequency"))
    print("")
    print("COLOR_PLUS_BS_SIGNAL:")
    print(flag("color_bs"))
    print("")
    print("COLOR_PLUS_NUMBER_SIGNAL:")
    print(flag("color_number"))
    print("")
    print("CALIBRATION:")
    print(f"mean_|gap|={b.get('CALIBRATION')} pp (lower better); detail in report")
    print("")
    print("COVERAGE:")
    print(f"{b.get('COVERAGE')}% full; selective thresholds in report")
    print("")
    print("SIGNAL_STATUS:")
    print(signal_status)
    print("")
    print("PRODUCTION_STATUS:")
    print("UNCHANGED")
    print("")
    print("REPORT:")
    print("exports/COLOR_SIGNAL_DEEP_RESEARCH_REPORT.md")
    print("")
    print(banner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
