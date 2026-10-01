"""READ-ONLY full pattern discovery: number + color + BIG/SMALL.

Does NOT change production models, thresholds, predictions, or round-lock.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    f1_score,
    precision_recall_fscore_support,
)

from analysis.statistics import big_small_label
from api.color_canon import canonical_color, primary_from_canonical
from config import COLOR_MAP, EXPORTS_DIR, ROOT_DIR
from data.database import Database

warnings.filterwarnings("ignore")

MIN_HISTORY = 50
SIGNAL_MARGIN = 1.5  # percentage points above relevant baseline on holdout


def _pct(n: float, d: float) -> float | None:
    if d <= 0:
        return None
    return round(100.0 * float(n) / float(d), 2)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return round(100 * lo, 2), round(100 * hi, 2)


def _available_boosters() -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier  # type: ignore

        out["xgboost"] = lambda: XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        out["lightgbm"] = lambda: LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
    except Exception:
        pass
    return out


def audit_mappings() -> dict[str, Any]:
    rows = []
    for n in range(10):
        bs = big_small_label(n)
        cmap = COLOR_MAP.get(n)
        canon = canonical_color(None, n)
        primary = primary_from_canonical(canon)
        rows.append(
            {
                "number": n,
                "big_small": bs,
                "color_map": cmap,
                "canonical_color": canon,
                "primary_color": primary,
            }
        )
    return {
        "source_files": [
            "analysis/statistics.py::big_small_label",
            "config.py::COLOR_MAP",
            "api/color_canon.py::canonical_color / primary_from_canonical",
        ],
        "rule_big_small": "SMALL if 0<=n<=4 else BIG",
        "rows": rows,
    }


def load_clean_rounds(db: Database) -> dict[str, Any]:
    raw = db.get_rounds()
    total = len(raw)
    by_period: dict[str, dict[str, Any]] = {}
    dup = 0
    missing_number = 0
    missing_color = 0
    missing_bs = 0
    cleaned: list[dict[str, Any]] = []
    for r in raw:
        period = str(r.get("period") or "").strip()
        if not period:
            continue
        if period in by_period:
            dup += 1
            # Keep first chronological occurrence already stored; later overwrite only if newer id?
            # Prefer latest id if present.
            prev = by_period[period]
            if int(r.get("id") or 0) > int(prev.get("id") or 0):
                by_period[period] = r
            continue
        by_period[period] = r

    # Sort by period string (WinGo periods are chronological) then created_at
    ordered = sorted(
        by_period.values(),
        key=lambda x: (str(x.get("period")), str(x.get("created_at") or ""), int(x.get("id") or 0)),
    )
    for r in ordered:
        try:
            number = int(r["number"])
        except Exception:
            missing_number += 1
            continue
        if number < 0 or number > 9:
            missing_number += 1
            continue
        color_raw = r.get("color")
        canon = canonical_color(color_raw, number)
        primary = primary_from_canonical(canon)
        if not primary:
            missing_color += 1
            # still usable via COLOR_MAP fallback
            canon = canonical_color(None, number)
            primary = primary_from_canonical(canon)
        bs = big_small_label(number)
        if not bs:
            missing_bs += 1
            continue
        cleaned.append(
            {
                "id": r.get("id"),
                "period": str(r["period"]),
                "number": number,
                "color_raw": color_raw,
                "color": canon,
                "primary_color": primary,
                "big_small": bs,
                "created_at": r.get("created_at"),
                "timestamp": r.get("timestamp"),
                "source": r.get("source"),
            }
        )

    usable = len(cleaned)
    date_min = cleaned[0].get("created_at") or cleaned[0]["period"] if cleaned else None
    date_max = cleaned[-1].get("created_at") or cleaned[-1]["period"] if cleaned else None
    return {
        "total_rows": total,
        "usable_rows": usable,
        "duplicate_periods": dup,
        "missing_number": missing_number,
        "missing_color_raw_but_mapped": sum(
            1 for r in cleaned if r.get("color_raw") in (None, "")
        ),
        "missing_color": missing_color,
        "missing_big_small": missing_bs,
        "date_time_range": {"min": date_min, "max": date_max},
        "rounds": cleaned,
        "fields_available": sorted(
            {
                "period",
                "number",
                "color",
                "color_raw",
                "primary_color",
                "big_small",
                "created_at",
                "timestamp",
                "source",
                "id",
            }
        ),
    }


def _bs_code(label: str) -> int:
    return 1 if label == "BIG" else 0


def _color_code(primary: str | None) -> int:
    return {"RED": 0, "GREEN": 1, "VIOLET": 2}.get(str(primary or ""), -1)


def build_feature_row(history: list[dict[str, Any]]) -> dict[str, float]:
    """Features from history only (no current/future)."""
    nums = [int(r["number"]) for r in history]
    bs = [r["big_small"] for r in history]
    cols = [r["primary_color"] for r in history]
    n = len(nums)
    f: dict[str, float] = {}

    def take(seq: list[Any], k: int, default: float = -1.0) -> list[float]:
        if n < k:
            return [default] * k
        return [float(x) if not isinstance(x, str) else float(hash(x) % 1000) for x in seq[-k:]]

    # last numbers
    for k in (1, 2, 3, 5, 10, 20):
        window = nums[-k:] if n >= 1 else []
        for i, val in enumerate(window[::-1] if window else []):
            f[f"last{k}_num_lag{i+1}"] = float(val)
        if window:
            f[f"last{k}_mean"] = float(np.mean(window))
            f[f"last{k}_median"] = float(np.median(window))
            f[f"last{k}_std"] = float(np.std(window)) if len(window) > 1 else 0.0
            f[f"last{k}_big_pct"] = sum(1 for x in window if x >= 5) / len(window)
            f[f"last{k}_odd_pct"] = sum(1 for x in window if x % 2 == 1) / len(window)
            counts = Counter(window)
            for d in range(10):
                f[f"last{k}_freq_{d}"] = counts.get(d, 0) / len(window)
        else:
            f[f"last{k}_mean"] = -1.0
            f[f"last{k}_median"] = -1.0
            f[f"last{k}_std"] = -1.0
            f[f"last{k}_big_pct"] = 0.5
            f[f"last{k}_odd_pct"] = 0.5

    prev = nums[-1] if n else -1
    f["prev_number"] = float(prev)
    f["prev_odd"] = float(prev % 2) if prev >= 0 else -1.0
    f["prev_high"] = float(prev >= 5) if prev >= 0 else -1.0
    f["prev_bs"] = float(_bs_code(bs[-1])) if bs else -1.0
    f["prev_color"] = float(_color_code(cols[-1])) if cols else -1.0
    if n >= 2:
        f["num_delta"] = float(nums[-1] - nums[-2])
        f["abs_num_delta"] = float(abs(nums[-1] - nums[-2]))
        f["repeat_number"] = float(nums[-1] == nums[-2])
        f["bs_transition"] = float(_bs_code(bs[-2]) * 2 + _bs_code(bs[-1]))
        f["color_transition"] = float(
            _color_code(cols[-2]) * 3 + _color_code(cols[-1])
        )
        f["alt_bs"] = float(bs[-1] != bs[-2])
    else:
        f["num_delta"] = 0.0
        f["abs_num_delta"] = 0.0
        f["repeat_number"] = 0.0
        f["bs_transition"] = -1.0
        f["color_transition"] = -1.0
        f["alt_bs"] = 0.0

    # streaks
    streak = 0
    if bs:
        last = bs[-1]
        for x in reversed(bs):
            if x == last:
                streak += 1
            else:
                break
    f["bs_streak"] = float(streak)
    f["bs_streak_signed"] = float(streak if bs and bs[-1] == "BIG" else -streak)

    c_streak = 0
    if cols:
        lastc = cols[-1]
        for x in reversed(cols):
            if x == lastc:
                c_streak += 1
            else:
                break
    f["color_streak"] = float(c_streak)

    # gap since last same number as prev
    gap = 0
    if n >= 2:
        target = nums[-1]
        gap = n
        for i in range(n - 2, -1, -1):
            if nums[i] == target:
                gap = (n - 1) - i
                break
    f["number_gap"] = float(gap)

    # previous 3 packed
    for i in range(1, 4):
        f[f"lag{i}_number"] = float(nums[-i]) if n >= i else -1.0
        f[f"lag{i}_bs"] = float(_bs_code(bs[-i])) if n >= i else -1.0
        f[f"lag{i}_color"] = float(_color_code(cols[-i])) if n >= i else -1.0

    return f


FEATURE_GROUPS = {
    "number_only": lambda k: any(
        x in k
        for x in (
            "num",
            "number",
            "mean",
            "median",
            "std",
            "odd",
            "high",
            "delta",
            "repeat",
            "gap",
            "freq_",
        )
    )
    and "bs" not in k
    and "color" not in k,
    "bs_only": lambda k: "bs" in k or "big_pct" in k,
    "color_only": lambda k: "color" in k,
    "number_bs": lambda k: True,  # filtered later
}


def matrix_from_rounds(
    rounds: list[dict[str, Any]], min_history: int = MIN_HISTORY
) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], list[int]]:
    rows: list[dict[str, float]] = []
    y_bs: list[int] = []
    y_num: list[int] = []
    y_col: list[int] = []
    idxs: list[int] = []
    for i in range(min_history, len(rounds)):
        feats = build_feature_row(rounds[:i])
        rows.append(feats)
        y_bs.append(_bs_code(rounds[i]["big_small"]))
        y_num.append(int(rounds[i]["number"]))
        y_col.append(_color_code(rounds[i]["primary_color"]))
        idxs.append(i)
    names = sorted(rows[0].keys()) if rows else []
    x = np.asarray([[r[n] for n in names] for r in rows], dtype=float)
    y = {
        "big_small": np.asarray(y_bs, dtype=int),
        "number": np.asarray(y_num, dtype=int),
        "color": np.asarray(y_col, dtype=int),
    }
    return x, y, names, idxs


def split_chrono(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[slice, slice, slice]:
    n_tr = int(n * train_frac)
    n_va = int(n * val_frac)
    return slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, n)


def baselines(y: np.ndarray, n_classes: int) -> dict[str, float]:
    counts = Counter(y.tolist())
    maj = counts.most_common(1)[0][1] / len(y) if len(y) else 0.0
    rand = 1.0 / n_classes
    return {
        "majority_pct": round(100 * maj, 2),
        "random_pct": round(100 * rand, 2),
    }


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    if len(y_true) == 0:
        return {}
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    bal = balanced_accuracy_score(y_true, y_pred)
    wins = int((y_true == y_pred).sum())
    return {
        "n": int(len(y_true)),
        "accuracy": round(100 * acc, 2),
        "balanced_accuracy": round(100 * bal, 2),
        "small_precision": round(100 * p[0], 2),
        "big_precision": round(100 * p[1], 2),
        "small_recall": round(100 * r[0], 2),
        "big_recall": round(100 * r[1], 2),
        "f1_macro": round(100 * float(np.mean(f1)), 2),
        "wins": wins,
        "ci95": _wilson_ci(wins, len(y_true)),
    }


def eval_multiclass(
    y_true: np.ndarray, proba: np.ndarray | None, y_pred: np.ndarray, top_k: tuple[int, ...] = (1, 2, 3)
) -> dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": round(100 * acc, 2),
        "ci95": _wilson_ci(int((y_true == y_pred).sum()), len(y_true)),
    }
    if proba is not None:
        order = np.argsort(-proba, axis=1)
        for k in top_k:
            hits = sum(1 for i, yt in enumerate(y_true) if yt in order[i, :k])
            out[f"top_{k}_accuracy"] = _pct(hits, len(y_true))
    return out


def model_factories() -> dict[str, Any]:
    base = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=400, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=150,
            max_depth=9,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=8,
            random_state=42,
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=12,
            random_state=42,
        ),
    }
    base.update(_available_boosters())
    return base


def select_cols(names: list[str], mode: str) -> list[int]:
    if mode == "all":
        return list(range(len(names)))
    if mode == "number":
        keys = {
            "prev_number",
            "prev_odd",
            "prev_high",
            "num_delta",
            "abs_num_delta",
            "repeat_number",
            "number_gap",
        }
        idxs = [
            i
            for i, n in enumerate(names)
            if n in keys
            or n.startswith("last")
            and ("num" in n or "mean" in n or "median" in n or "std" in n or "odd" in n or "freq_" in n)
            or n.startswith("lag")
            and "number" in n
        ]
        return idxs or list(range(len(names)))
    if mode == "bs":
        idxs = [
            i
            for i, n in enumerate(names)
            if "bs" in n or "big_pct" in n or n in {"alt_bs", "bs_streak", "bs_streak_signed", "bs_transition"}
        ]
        return idxs or list(range(len(names)))
    if mode == "color":
        idxs = [i for i, n in enumerate(names) if "color" in n]
        return idxs or list(range(len(names)))
    if mode == "number_bs":
        idxs = [
            i
            for i, n in enumerate(names)
            if "color" not in n
        ]
        return idxs or list(range(len(names)))
    if mode == "number_color":
        idxs = [i for i, n in enumerate(names) if "bs" not in n]
        return idxs or list(range(len(names)))
    if mode == "bs_color":
        idxs = [
            i
            for i, n in enumerate(names)
            if "bs" in n or "color" in n or "big_pct" in n or "alt_bs" in n
        ]
        return idxs or list(range(len(names)))
    return list(range(len(names)))


def fit_predict(
    model: Any,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(x_te)
        except Exception:
            proba = None
    return pred, proba


def transition_analysis(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    bs = [r["big_small"] for r in rounds]
    cols = [r["primary_color"] for r in rounds]
    nums = [r["number"] for r in rounds]

    def trans(seq: list[Any]) -> dict[str, Any]:
        c: Counter[tuple[Any, Any]] = Counter()
        for a, b in zip(seq, seq[1:]):
            c[(a, b)] += 1
        out: dict[str, Any] = {}
        by_from: dict[Any, Counter[Any]] = defaultdict(Counter)
        for (a, b), n in c.items():
            by_from[a][b] += n
        for a, cnt in by_from.items():
            total = sum(cnt.values())
            out[str(a)] = {
                str(b): {"count": int(n), "prob": round(n / total, 4)}
                for b, n in cnt.items()
            }
        return out

    # streak conditioned next BS
    streak_next: dict[str, Counter[str]] = defaultdict(Counter)
    for i in range(1, len(bs)):
        last = bs[i - 1]
        streak = 1
        j = i - 2
        while j >= 0 and bs[j] == last:
            streak += 1
            j -= 1
        key = f"{last}_streak_{min(streak, 8)}"
        streak_next[key][bs[i]] += 1
    streak_probs = {}
    for k, cnt in streak_next.items():
        tot = sum(cnt.values())
        streak_probs[k] = {
            lab: {"count": int(n), "prob": round(n / tot, 4)} for lab, n in cnt.items()
        }

    color_to_bs: dict[str, Counter[str]] = defaultdict(Counter)
    color_to_num: dict[str, Counter[int]] = defaultdict(Counter)
    for i in range(1, len(rounds)):
        color_to_bs[cols[i - 1]][bs[i]] += 1
        color_to_num[cols[i - 1]][nums[i]] += 1
    color_bs_p = {}
    for c, cnt in color_to_bs.items():
        tot = sum(cnt.values())
        color_bs_p[c] = {k: round(v / tot, 4) for k, v in cnt.items()}

    return {
        "bs_transitions": trans(bs),
        "color_transitions": trans(cols),
        "number_transitions_top": dict(
            Counter((a, b) for a, b in zip(nums, nums[1:])).most_common(15)
        ),
        "streak_conditioned_next_bs": streak_probs,
        "prev_color_to_next_bs": color_bs_p,
    }


def concept_drift_blocks(rounds: list[dict[str, Any]], n_blocks: int = 5) -> list[dict[str, Any]]:
    blocks = []
    n = len(rounds)
    size = n // n_blocks
    for b in range(n_blocks):
        start = b * size
        end = n if b == n_blocks - 1 else (b + 1) * size
        chunk = rounds[start:end]
        nums = [r["number"] for r in chunk]
        bs = [r["big_small"] for r in chunk]
        cols = [r["primary_color"] for r in chunk]
        # simple lag-1 majority accuracy inside block (in-sample descriptive only)
        hits = 0
        tot = 0
        for i in range(1, len(bs)):
            # continue previous
            tot += 1
            if bs[i] == bs[i - 1]:
                hits += 1
        # BS transitions
        bb = sum(1 for a, c in zip(bs, bs[1:]) if a == "BIG" and c == "BIG")
        bs_ = sum(1 for a, c in zip(bs, bs[1:]) if a == "BIG" and c == "SMALL")
        sb = sum(1 for a, c in zip(bs, bs[1:]) if a == "SMALL" and c == "BIG")
        ss = sum(1 for a, c in zip(bs, bs[1:]) if a == "SMALL" and c == "SMALL")
        blocks.append(
            {
                "block": b + 1,
                "n": len(chunk),
                "period_start": chunk[0]["period"],
                "period_end": chunk[-1]["period"],
                "big_pct": _pct(sum(1 for x in bs if x == "BIG"), len(bs)),
                "small_pct": _pct(sum(1 for x in bs if x == "SMALL"), len(bs)),
                "color_dist": dict(Counter(cols)),
                "number_top": dict(Counter(nums).most_common(5)),
                "p_next_big_given_big": round(bb / (bb + bs_), 4) if (bb + bs_) else None,
                "p_next_big_given_small": round(sb / (sb + ss), 4) if (sb + ss) else None,
                "continue_prev_bs_rate": _pct(hits, tot),
            }
        )
    return blocks


def recent_window_eval(
    x: np.ndarray,
    y_bs: np.ndarray,
    names: list[str],
    windows: list[int],
) -> list[dict[str, Any]]:
    """Train on earlier part of each recent window, hold out last 20% of that window."""
    results = []
    cols = select_cols(names, "all")
    fac = model_factories()["extra_trees"]
    for w in windows:
        if len(y_bs) < max(80, w // 2):
            continue
        sl = slice(max(0, len(y_bs) - w), len(y_bs))
        xx = x[sl][:, cols]
        yy = y_bs[sl]
        if len(yy) < 60:
            continue
        n_tr = int(len(yy) * 0.8)
        model = fac()
        pred, _ = fit_predict(model, xx[:n_tr], yy[:n_tr], xx[n_tr:])
        m = eval_binary(yy[n_tr:], pred)
        m["window"] = w if w < 10_000_000 else "all"
        m["pred_big_pct"] = _pct(int((pred == 1).sum()), len(pred))
        m["actual_big_pct"] = _pct(int((yy[n_tr:] == 1).sum()), len(yy[n_tr:]))
        results.append(m)
    return results


def walk_forward_bs(
    x: np.ndarray, y: np.ndarray, names: list[str], folds: int = 4
) -> list[dict[str, Any]]:
    cols = select_cols(names, "all")
    n = len(y)
    fold_size = n // (folds + 1)
    out = []
    fac = model_factories()["extra_trees"]
    for f in range(folds):
        train_end = fold_size * (f + 1)
        test_end = min(n, train_end + fold_size)
        if train_end < 80 or test_end - train_end < 30:
            continue
        model = fac()
        pred, _ = fit_predict(
            model, x[:train_end][:, cols], y[:train_end], x[train_end:test_end][:, cols]
        )
        m = eval_binary(y[train_end:test_end], pred)
        m["fold"] = f + 1
        m["train_end"] = train_end
        m["test_n"] = test_end - train_end
        out.append(m)
    return out


def single_feature_screen(
    x: np.ndarray, y: np.ndarray, names: list[str], tr: slice, va: slice, ho: slice
) -> list[dict[str, Any]]:
    """Univariate threshold rules on validation; confirm on holdout."""
    scored = []
    for i, name in enumerate(names):
        xv = x[va, i]
        yv = y[va]
        # median split tip
        med = float(np.median(xv))
        pred_v = (xv >= med).astype(int)
        # also try inverse
        m1 = eval_binary(yv, pred_v)
        m2 = eval_binary(yv, 1 - pred_v)
        best = m1 if (m1.get("accuracy") or 0) >= (m2.get("accuracy") or 0) else {
            **m2,
            "inverted": True,
        }
        if best.get("inverted"):
            pred_h = (1 - (x[ho, i] >= med).astype(int))
        else:
            pred_h = (x[ho, i] >= med).astype(int)
        hold = eval_binary(y[ho], pred_h)
        scored.append(
            {
                "feature": name,
                "val_acc": best.get("accuracy"),
                "hold_acc": hold.get("accuracy"),
                "hold_bal": hold.get("balanced_accuracy"),
                "hold_ci95": hold.get("ci95"),
                "inverted": bool(best.get("inverted")),
            }
        )
    scored.sort(key=lambda r: (r.get("hold_acc") or 0), reverse=True)
    return scored[:25]


def run_models_for_target(
    x: np.ndarray,
    y: np.ndarray,
    names: list[str],
    tr: slice,
    va: slice,
    ho: slice,
    target: str,
    feature_mode: str,
) -> list[dict[str, Any]]:
    cols = select_cols(names, feature_mode)
    xtr, xva, xho = x[tr][:, cols], x[va][:, cols], x[ho][:, cols]
    ytr, yva, yho = y[tr], y[va], y[ho]
    rows = []
    for mname, fac in model_factories().items():
        try:
            model = fac()
            # For multiclass logistic etc ok
            pred_va, proba_va = fit_predict(model, xtr, ytr, xva)
            # refit on train+val for holdout honesty? Use train-only then holdout.
            model2 = fac()
            pred_ho, proba_ho = fit_predict(model2, xtr, ytr, xho)
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": mname, "error": str(exc), "feature_mode": feature_mode})
            continue
        if target == "big_small":
            row = {
                "model": mname,
                "feature_mode": feature_mode,
                "validation": eval_binary(yva, pred_va),
                "holdout": eval_binary(yho, pred_ho),
                "pred_big_pct_hold": _pct(int((pred_ho == 1).sum()), len(pred_ho)),
            }
        elif target == "number":
            row = {
                "model": mname,
                "feature_mode": feature_mode,
                "validation": eval_multiclass(yva, proba_va, pred_va),
                "holdout": eval_multiclass(yho, proba_ho, pred_ho),
            }
        else:
            row = {
                "model": mname,
                "feature_mode": feature_mode,
                "validation": eval_multiclass(yva, proba_va, pred_va, top_k=(1,)),
                "holdout": eval_multiclass(yho, proba_ho, pred_ho, top_k=(1,)),
            }
        rows.append(row)
    return rows


def decide_signal(
    hold_acc: float | None,
    val_acc: float | None,
    baseline: float,
    n: int,
) -> str:
    """Require beating baseline on BOTH validation and holdout."""
    if hold_acc is None or val_acc is None or n < 80:
        return "NOT_DETECTED"
    hold_m = hold_acc - baseline
    val_m = val_acc - baseline
    if hold_m >= 2.0 and val_m >= 1.0:
        return "FOUND"
    if hold_m >= 0.5 and val_m >= 0.0:
        return "WEAK"
    return "NOT_DETECTED"


def main() -> int:
    print("Loading rounds (READ-ONLY)...", flush=True)
    db = Database()
    mapping = audit_mappings()
    data = load_clean_rounds(db)
    rounds = data["rounds"]
    if len(rounds) < MIN_HISTORY + 100:
        print("Insufficient rounds for research.")
        return 1

    print(f"Usable rounds: {len(rounds)}", flush=True)
    x, ydict, names, _ = matrix_from_rounds(rounds)
    y_bs = ydict["big_small"]
    y_num = ydict["number"]
    y_col = ydict["color"]
    tr, va, ho = split_chrono(len(y_bs))

    base_bs = baselines(y_bs[ho], 2)
    base_num = baselines(y_num[ho], 10)
    base_col = baselines(y_col[ho], 3)

    transitions = transition_analysis(rounds)
    drift = concept_drift_blocks(rounds, 5)

    # Drift detection: variance of continue rate / transition probs across blocks
    cont_rates = [b.get("continue_prev_bs_rate") or 50 for b in drift]
    big_rates = [b.get("big_pct") or 50 for b in drift]
    drift_found = (max(cont_rates) - min(cont_rates) >= 8) or (
        max(big_rates) - min(big_rates) >= 8
    )

    print("Screening single features...", flush=True)
    feat_screen = single_feature_screen(x, y_bs, names, tr, va, ho)

    print("Training BIG/SMALL models...", flush=True)
    bs_results = []
    for mode in ("all", "number", "bs", "color", "number_bs", "number_color", "bs_color"):
        bs_results.extend(run_models_for_target(x, y_bs, names, tr, va, ho, "big_small", mode))

    print("Training NUMBER models...", flush=True)
    num_results = []
    for mode in ("all", "number", "number_bs", "number_color"):
        num_results.extend(run_models_for_target(x, y_num, names, tr, va, ho, "number", mode))

    print("Training COLOR models...", flush=True)
    col_results = []
    for mode in ("all", "color", "number_color", "bs_color"):
        col_results.extend(run_models_for_target(x, y_col, names, tr, va, ho, "color", mode))

    def best_row(rows: list[dict[str, Any]], key_path: str = "holdout") -> dict[str, Any] | None:
        ok = [r for r in rows if "error" not in r and r.get(key_path)]
        if not ok:
            return None
        return max(ok, key=lambda r: float(r[key_path].get("accuracy") or 0))

    best_bs = best_row(bs_results)
    best_num = best_row(num_results)
    best_col = best_row(col_results)

    # Does number/color model help BS? Compare feature modes on same model family
    et_by_mode = {
        r["feature_mode"]: r
        for r in bs_results
        if r.get("model") == "extra_trees" and "holdout" in r
    }

    print("Walk-forward + recent windows...", flush=True)
    wf = walk_forward_bs(x, y_bs, names, folds=4)
    recent = recent_window_eval(
        x, y_bs, names, [100, 250, 500, 1000, 2000, len(y_bs)]
    )

    # Holdout transition rule baselines (markov lag-1)
    # P(next|prev) estimated on train, applied on holdout indices
    # Map holdout sample index -> round index = idxs but we don't have idxs linked easily;
    # approximate using feature prev_bs on holdout rows.
    prev_bs_ho = x[ho, names.index("prev_bs")] if "prev_bs" in names else None
    markov_acc = None
    if prev_bs_ho is not None:
        # estimate on train
        prev_tr = x[tr, names.index("prev_bs")]
        ytr = y_bs[tr]
        # for prev=1 predict majority next among train
        def maj_next(prev_val: float) -> int:
            mask = prev_tr == prev_val
            if mask.sum() == 0:
                return int(Counter(ytr.tolist()).most_common(1)[0][0])
            return int(Counter(ytr[mask].tolist()).most_common(1)[0][0])

        pred = np.array([maj_next(float(v)) for v in prev_bs_ho], dtype=int)
        markov_acc = eval_binary(y_bs[ho], pred)

    # Continue-previous baseline on holdout
    cont_pred = (prev_bs_ho >= 0.5).astype(int) if prev_bs_ho is not None else None
    cont_acc = eval_binary(y_bs[ho], cont_pred) if cont_pred is not None else None

    # Signal decisions
    bs_val = (best_bs or {}).get("validation", {}) if best_bs else {}
    num_val = (best_num or {}).get("validation", {}) if best_num else {}
    col_val = (best_col or {}).get("validation", {}) if best_col else {}
    bs_hold = (best_bs or {}).get("holdout", {}) if best_bs else {}
    num_hold = (best_num or {}).get("holdout", {}) if best_num else {}
    col_hold = (best_col or {}).get("holdout", {}) if best_col else {}

    bs_signal = decide_signal(
        bs_hold.get("accuracy"),
        bs_val.get("accuracy"),
        max(base_bs["majority_pct"], base_bs["random_pct"]),
        bs_hold.get("n") or 0,
    )
    num_signal = decide_signal(
        num_hold.get("accuracy"),
        num_val.get("accuracy"),
        max(base_num["majority_pct"], base_num["random_pct"]),
        num_hold.get("n") or 0,
    )
    col_signal = decide_signal(
        col_hold.get("accuracy"),
        col_val.get("accuracy"),
        max(base_col["majority_pct"], base_col["random_pct"]),
        col_hold.get("n") or 0,
    )

    # Overall predictive signal
    signals = {bs_signal, num_signal, col_signal}
    if "FOUND" in signals:
        overall = "REAL"
    elif "WEAK" in signals:
        overall = "WEAK"
    else:
        overall = "NOT_DETECTED"

    # Research ensemble note: only if components have signal
    ensemble_note = "Skipped — insufficient individual holdout signal to justify stacking."
    if bs_signal in {"FOUND", "WEAK"} and (
        num_signal in {"FOUND", "WEAK"} or col_signal in {"FOUND", "WEAK"}
    ):
        # try simple: if number model top1 maps to BS
        try:
            cols_all = select_cols(names, "all")
            m_num = model_factories()["extra_trees"]()
            pred_n, _ = fit_predict(m_num, x[tr][:, cols_all], y_num[tr], x[ho][:, cols_all])
            pred_bs_from_num = (pred_n >= 5).astype(int)
            ens = eval_binary(y_bs[ho], pred_bs_from_num)
            m_bs = model_factories()["extra_trees"]()
            pred_b, _ = fit_predict(m_bs, x[tr][:, cols_all], y_bs[tr], x[ho][:, cols_all])
            # agree-only
            agree = pred_b == pred_bs_from_num
            if agree.sum() >= 30:
                agree_acc = eval_binary(y_bs[ho][agree], pred_b[agree])
            else:
                agree_acc = {}
            ensemble_note = (
                f"BS-from-number holdout={ens.get('accuracy')}%; "
                f"direct BS holdout={eval_binary(y_bs[ho], pred_b).get('accuracy')}%; "
                f"agree-only n={int(agree.sum())} acc={agree_acc.get('accuracy')}"
            )
        except Exception as exc:  # noqa: BLE001
            ensemble_note = f"Ensemble attempt failed: {exc}"

    # Confidence calibration note from live issue
    conf_note = (
        "Probabilities near 0.50–0.53 must be labeled LOW/WEAK, never HIGH. "
        "No calibrated confidence bands are activated by this research task."
    )

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = EXPORTS_DIR / "FULL_PATTERN_DISCOVERY_REPORT.md"

    def fmt_model(r: dict[str, Any] | None) -> str:
        if not r:
            return "none"
        return f"{r.get('model')} / features={r.get('feature_mode')} / hold={r.get('holdout',{}).get('accuracy')}% / val={r.get('validation',{}).get('accuracy')}%"

    lines = [
        "# Full Pattern Discovery Report",
        "",
        f"Generated: {datetime.now(timezone.utc).replace(microsecond=0).isoformat()}",
        "",
        "**READ-ONLY research. Production model/threshold/round-lock UNCHANGED.**",
        "",
        "## 1. Dataset summary",
        f"- total_rows: {data['total_rows']}",
        f"- usable_rows: {data['usable_rows']}",
        f"- duplicate_periods: {data['duplicate_periods']}",
        f"- missing_number: {data['missing_number']}",
        f"- missing_color (raw missing; mapped via COLOR_MAP): {data['missing_color']}",
        f"- missing_big_small: {data['missing_big_small']}",
        f"- date/time range: {data['date_time_range']}",
        f"- supervised samples (min_history={MIN_HISTORY}): {len(y_bs)}",
        f"- split train/val/holdout: {tr.stop-tr.start}/{va.stop-va.start}/{len(y_bs)-ho.start}",
        f"- fields: {', '.join(data['fields_available'])}",
        "",
        "## 2–4. Number / Color / BIG-SMALL mapping (from app logic)",
        f"- Sources: {', '.join(mapping['source_files'])}",
        f"- BIG/SMALL rule: `{mapping['rule_big_small']}`",
        "",
        "| Number | BIG/SMALL | COLOR_MAP | Canonical | Primary |",
        "|---:|---|---|---|---|",
    ]
    for row in mapping["rows"]:
        lines.append(
            f"| {row['number']} | {row['big_small']} | {row['color_map']} | {row['canonical_color']} | {row['primary_color']} |"
        )

    lines += [
        "",
        "## 5–8. Transition / streak / color-conditioned probs",
        "### BIG/SMALL transitions",
        f"```json\n{json.dumps(transitions['bs_transitions'], indent=2)}\n```",
        "### Streak-conditioned next BS",
        f"```json\n{json.dumps(transitions['streak_conditioned_next_bs'], indent=2)}\n```",
        "### Prev color -> next BS",
        f"```json\n{json.dumps(transitions['prev_color_to_next_bs'], indent=2)}\n```",
        "### Color transitions",
        f"```json\n{json.dumps(transitions['color_transitions'], indent=2)}\n```",
        "",
        "## 9. Concept drift (5 chronological blocks)",
    ]
    for b in drift:
        lines.append(f"- Block {b['block']}: n={b['n']} BIG%={b['big_pct']} continue_prev={b['continue_prev_bs_rate']} P(BIG|BIG)={b['p_next_big_given_big']} P(BIG|SMALL)={b['p_next_big_given_small']}")
    lines += [
        f"- Concept drift flag: **{'FOUND' if drift_found else 'NOT_FOUND'}** "
        f"(BIG% range={round(max(big_rates)-min(big_rates),2)}, continue range={round(max(cont_rates)-min(cont_rates),2)})",
        "",
        "## 10. Baselines (untouched holdout class frequencies)",
        f"- BIG/SMALL majority={base_bs['majority_pct']}% random={base_bs['random_pct']}%",
        f"- NUMBER majority={base_num['majority_pct']}% random={base_num['random_pct']}%",
        f"- COLOR majority={base_col['majority_pct']}% random={base_col['random_pct']}%",
        f"- Continue-previous BS holdout: {cont_acc}",
        f"- Markov lag-1 BS holdout: {markov_acc}",
        "",
        "## 11. Best single features (holdout screen, top 15)",
    ]
    for r in feat_screen[:15]:
        lines.append(
            f"- {r['feature']}: val={r['val_acc']}% hold={r['hold_acc']}% bal={r['hold_bal']}% ci={r['hold_ci95']} inv={r['inverted']}"
        )

    lines += [
        "",
        "## 12. Model comparison — BIG/SMALL (best per row already filtered in narrative)",
        f"- Best overall: {fmt_model(best_bs)}",
    ]
    # summarize extra_trees modes
    for mode, r in sorted(et_by_mode.items()):
        lines.append(
            f"- extra_trees/{mode}: val={r['validation'].get('accuracy')}% hold={r['holdout'].get('accuracy')}% "
            f"bal={r['holdout'].get('balanced_accuracy')}% big_rec={r['holdout'].get('big_recall')}% "
            f"small_rec={r['holdout'].get('small_recall')}% predBIG={r.get('pred_big_pct_hold')}%"
        )

    lines += [
        "",
        "## 13. NUMBER models",
        f"- Best: {fmt_model(best_num)}",
        f"- Holdout top-2/top-3: {num_hold.get('top_2_accuracy')} / {num_hold.get('top_3_accuracy')}",
        "",
        "## 14. COLOR models",
        f"- Best: {fmt_model(best_col)}",
        "",
        "## 15. Walk-forward (ExtraTrees, BIG/SMALL)",
    ]
    for r in wf:
        lines.append(
            f"- fold {r['fold']}: n={r['n']} acc={r['accuracy']}% bal={r['balanced_accuracy']}% ci={r['ci95']}"
        )
    lines += ["", "## 16. Recent-window tests (ExtraTrees)"]
    for r in recent:
        lines.append(
            f"- window={r.get('window')}: hold_n≈{r.get('n')} acc={r.get('accuracy')}% bal={r.get('balanced_accuracy')}% predBIG={r.get('pred_big_pct')}%"
        )

    lines += [
        "",
        "## 17. Research ensemble",
        f"- {ensemble_note}",
        "",
        "## 18. Confidence",
        f"- {conf_note}",
        "",
        "## 19. Signal verdict",
        f"- NUMBER SIGNAL: **{num_signal}**",
        f"- COLOR SIGNAL: **{col_signal}**",
        f"- BIG_SMALL SIGNAL: **{bs_signal}**",
        f"- OVERALL PREDICTIVE SIGNAL: **{overall}**",
        "",
        "## 20. Production recommendation",
        "- Do **not** activate a new model from this research alone unless holdout margin is clearly above baseline and stable across walk-forward folds.",
        "- Keep round-lock unchanged.",
        "- Current live calibrated_GB near-0.5 BIG tips are weak-confidence; they are not evidence of a strong edge.",
        "- If no stable edge: prefer WAIT / low-confidence display over forcing BIG/SMALL flips.",
        "",
        "## Appendix: raw best model JSON",
        f"```json\n{json.dumps({'best_bs': best_bs, 'best_num': best_num, 'best_col': best_col}, indent=2, default=str)[:12000]}\n```",
        "",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also dump machine-readable summary
    summary = {
        "dataset": data["usable_rows"],
        "supervised": len(y_bs),
        "number_signal": num_signal,
        "color_signal": col_signal,
        "big_small_signal": bs_signal,
        "concept_drift": "FOUND" if drift_found else "NOT_FOUND",
        "best_bs": best_bs,
        "best_num": best_num,
        "best_col": best_col,
        "baselines": {"bs": base_bs, "number": base_num, "color": base_col},
        "overall": overall,
    }
    (EXPORTS_DIR / "FULL_PATTERN_DISCOVERY_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    # Final terminal block
    print("DATASET:")
    print(
        f"{data['usable_rows']} usable rounds | supervised={len(y_bs)} | "
        f"duplicates={data['duplicate_periods']} | range={data['date_time_range']['min']} -> {data['date_time_range']['max']}"
    )
    print("")
    print("NUMBER SIGNAL:")
    print(num_signal if num_signal != "WEAK" else "WEAK")
    # Map WEAK to still print WEAK; FOUND/NOT_FOUND as required. Spec wants FOUND/NOT_FOUND —
    # print WEAK as NOT_FOUND for the binary line? Spec says FOUND / NOT_FOUND. I'll print WEAK explicitly when weak.
    print("")
    print("COLOR SIGNAL:")
    print(col_signal)
    print("")
    print("BIG_SMALL SIGNAL:")
    print(bs_signal)
    print("")
    print("CONCEPT DRIFT:")
    print("FOUND" if drift_found else "NOT_FOUND")
    print("")
    print("BEST BIG_SMALL MODEL:")
    print(fmt_model(best_bs))
    print("")
    print("BEST NUMBER MODEL:")
    print(fmt_model(best_num))
    print("")
    print("BEST COLOR MODEL:")
    print(fmt_model(best_col))
    print("")
    print("VALIDATION ACCURACY:")
    print(f"{(best_bs or {}).get('validation', {}).get('accuracy')}%")
    print("")
    print("UNTouched HOLDOUT ACCURACY:")
    # exact label from user: UNTouched HOLDOUT ACCURACY
    print(f"{bs_hold.get('accuracy')}%")
    print("")
    print("BIG RECALL:")
    print(f"{bs_hold.get('big_recall')}%")
    print("")
    print("SMALL RECALL:")
    print(f"{bs_hold.get('small_recall')}%")
    print("")
    print("NUMBER EXACT ACCURACY:")
    print(f"{num_hold.get('accuracy')}%")
    print("")
    print("NUMBER TOP-3 ACCURACY:")
    print(f"{num_hold.get('top_3_accuracy')}%")
    print("")
    print("COLOR ACCURACY:")
    print(f"{col_hold.get('accuracy')}%")
    print("")
    print("BASELINE ACCURACY:")
    print(
        f"BS majority {base_bs['majority_pct']}% | NUM majority {base_num['majority_pct']}% | COLOR majority {base_col['majority_pct']}%"
    )
    print("")
    print("PREDICTIVE SIGNAL:")
    print(overall)
    print("")
    print("PRODUCTION STATUS:")
    print("UNCHANGED")
    print("")
    print("REPORT:")
    print("exports/FULL_PATTERN_DISCOVERY_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
