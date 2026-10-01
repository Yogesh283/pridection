"""Chronological walk-forward comparison of existing tip methods (no leakage).

Usage:
  python -m tools.walkforward_compare
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.big_small_analyze import analyze_then_predict
from analysis.statistics import big_small_label
from api.client import primary_color
from data.database import Database
from models.ensemble import EnsemblePredictor
from models.predictor import (
    FrequencyModel,
    MajorityWindowModel,
    MarkovModel,
    PatternModel,
    RecentWeightedModel,
)


def _pct(ok: int, n: int) -> float:
    return round(100.0 * ok / n, 2) if n else 0.0


def _wilson_low(ok: int, n: int, z: float = 1.96) -> float:
    """Lower Wilson score bound for binomial accuracy (conservative)."""
    if n <= 0:
        return 0.0
    p = ok / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round(100.0 * max(0.0, (centre - margin) / denom), 2)


def load_rounds() -> list[dict[str, Any]]:
    rounds = Database().get_rounds()
    if len(rounds) < 80:
        raise SystemExit(f"need more rounds in DB, got {len(rounds)}")
    return rounds


def tip_from_dist(bs: dict[str, float]) -> str | None:
    """Majority-style tip; None on exact probability tie (= WAIT)."""
    big = float(bs.get("BIG", 0))
    small = float(bs.get("SMALL", 0))
    if big == small:
        return None
    return "BIG" if big > small else "SMALL"


def eval_method(
    name: str,
    rounds: list[dict[str, Any]],
    predict_fn: Callable[[list[dict[str, Any]]], dict[str, Any] | None],
    *,
    min_history: int = 80,
) -> dict[str, Any]:
    """
    Walk-forward: for each i, predict using rounds[:i] only, score vs rounds[i].
    predict_fn returns dict with optional:
      big_small, number, color, skip (bool), confidence (0..1)
    """
    bs_ok = bs_n = 0
    num_ok = num_n = 0
    col_ok = col_n = 0
    skipped = 0
    conf_sum = 0.0
    conf_n = 0
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # ok, n

    for i in range(min_history, len(rounds)):
        history = rounds[:i]
        actual = rounds[i]
        out = predict_fn(history)
        if out is None:
            skipped += 1
            continue
        if out.get("skip"):
            skipped += 1
            continue

        act_bs = big_small_label(int(actual["number"]))
        act_num = int(actual["number"])
        act_col = primary_color(actual.get("color"))

        pred_bs = out.get("big_small")
        if pred_bs in ("BIG", "SMALL"):
            hit = int(pred_bs == act_bs)
            bs_ok += hit
            bs_n += 1
            conf = out.get("confidence")
            if conf is not None:
                conf_sum += float(conf)
                conf_n += 1
                if float(conf) < 0.55:
                    b = "conf_<55"
                elif float(conf) < 0.62:
                    b = "conf_55_62"
                else:
                    b = "conf_>=62"
                buckets[b][0] += hit
                buckets[b][1] += 1

        pred_num = out.get("number")
        if pred_num is not None:
            num_ok += int(int(pred_num) == act_num)
            num_n += 1

        pred_col = out.get("color")
        if pred_col and act_col:
            col_ok += int(str(pred_col) == act_col)
            col_n += 1

    total_slots = len(rounds) - min_history
    return {
        "name": name,
        "bs_accuracy": _pct(bs_ok, bs_n),
        "bs_wilson_low": _wilson_low(bs_ok, bs_n),
        "bs_ok": bs_ok,
        "bs_n": bs_n,
        "bs_wrong": bs_n - bs_ok,
        "number_accuracy": _pct(num_ok, num_n),
        "number_n": num_n,
        "color_accuracy": _pct(col_ok, col_n),
        "color_n": col_n,
        "skipped": skipped,
        "coverage_pct": _pct(bs_n, total_slots),
        "avg_confidence": round(conf_sum / conf_n, 4) if conf_n else None,
        "confidence_buckets": {
            k: {"accuracy": _pct(v[0], v[1]), "n": v[1], "ok": v[0]}
            for k, v in sorted(buckets.items())
        },
        "baseline_bs": 50.0,
        "lift_vs_50": round(_pct(bs_ok, bs_n) - 50.0, 2) if bs_n else 0.0,
    }


def main() -> int:
    rounds = load_rounds()
    print(f"rounds={len(rounds)} min_history=80 scored_slots={len(rounds)-80}")

    maj = MajorityWindowModel()
    freq = FrequencyModel()
    markov = MarkovModel()
    recent = RecentWeightedModel()
    pattern = PatternModel()
    ens = EnsemblePredictor()
    # Train ML once on earliest history only (no future).
    if len(rounds) > 280:
        ens.maybe_train_ml(rounds[:200])

    def _from_model(out):
        tip = tip_from_dist(out["big_small"])
        if tip is None:
            return {"skip": True, "confidence": 0.5}
        return {
            "big_small": tip,
            "number": max(out["numbers"], key=out["numbers"].get),
            "color": max(out["colors"], key=out["colors"].get),
            "confidence": float(out["big_small"][tip]),
        }

    def always_maj(h):
        return _from_model(maj.predict(h))

    def always_freq(h):
        return _from_model(freq.predict(h))

    def always_markov(h):
        return _from_model(markov.predict(h))

    def always_recent(h):
        return _from_model(recent.predict(h))

    def always_pattern(h):
        return _from_model(pattern.predict(h))

    def follow(h):
        last = big_small_label(int(h[-1]["number"]))
        return {"big_small": last, "number": int(h[-1]["number"]), "confidence": 0.5}

    def flip(h):
        last = big_small_label(int(h[-1]["number"]))
        tip = "BIG" if last == "SMALL" else "SMALL"
        return {"big_small": tip, "confidence": 0.5}

    def last10_current(h):
        tip = analyze_then_predict(h)["tip"]
        return {
            "big_small": tip.get("big_small"),
            "number": tip.get("number"),
            "skip": bool(tip.get("skip")),
            "confidence": max(float(v) for v in (tip.get("probs") or {"x": 0.5}).values()),
        }

    def ensemble_blend_no_override(h):
        """Use ensemble distributions WITHOUT last10 _pick_big_small override."""
        # Temporarily use number focus path? Better: call internals.
        # Predict with a monkey-patch: use focus that still overrides...
        # So we reconstruct tip from blend by calling base models.
        weights = ens.refresh_adaptive_weights(h, focus="big_small")
        outputs = {name: m.predict(h) for name, m in ens._base_models().items()}
        ml_out = ens.ml.predict(h)
        if ml_out:
            outputs["ml"] = ml_out
        from models.ensemble import _combine_distributions, _normalize
        from config import ENSEMBLE_WEIGHTS

        rows = []
        for name, out in outputs.items():
            dist = out.get("big_small")
            if not dist:
                continue
            w = weights.get(name, ENSEMBLE_WEIGHTS.get(name, 1.0))
            if w <= 0:
                continue
            rows.append((name, dist, w))
        bs = _combine_distributions(rows)
        tip = tip_from_dist(bs)
        n_rows = []
        for name, out in outputs.items():
            w = weights.get(name, ENSEMBLE_WEIGHTS.get(name, 1.0))
            if w <= 0:
                continue
            n_rows.append((name, out["numbers"], w))
        nums = _combine_distributions(n_rows)
        cols = _combine_distributions(
            [
                (name, out["colors"], weights.get(name, 1.0))
                for name, out in outputs.items()
                if weights.get(name, 1.0) > 0
            ]
        )
        if tip is None:
            return {
                "skip": True,
                "number": max(nums, key=nums.get) if nums else None,
                "color": max(cols, key=cols.get) if cols else None,
                "confidence": 0.5,
            }
        return {
            "big_small": tip,
            "number": max(nums, key=nums.get) if nums else None,
            "color": max(cols, key=cols.get) if cols else None,
            "confidence": float(bs.get(tip, 0.5)),
        }

    def ensemble_selective(h, min_margin: float = 0.08):
        out = ensemble_blend_no_override(h)
        conf = float(out["confidence"])
        margin = abs(2 * conf - 1.0)  # |p - (1-p)|
        if margin < min_margin:
            return {**out, "skip": True}
        return out

    def live_majority_w8_path(h):
        """Exact current live path: majority_w8 with WAIT on tie."""
        res = ens.predict(h, train_ml=False, focus="big_small")
        if not res:
            return None
        return {
            "big_small": None if res.get("skip_tip") else res.get("top_big_small"),
            "number": res.get("top_number"),
            "color": res.get("top_color"),
            "skip": bool(res.get("skip_tip")),
            "confidence": float(res.get("big_small_probability") or 0.5),
        }

    def ml_only(h):
        out = ens.ml.predict(h)
        if not out:
            return {"skip": True}
        tip = tip_from_dist(out["big_small"])
        return {
            "big_small": tip,
            "number": max(out["numbers"], key=out["numbers"].get),
            "color": max(out["colors"], key=out["colors"].get),
            "confidence": float(out["big_small"][tip]),
        }

    methods: list[tuple[str, Callable]] = [
        ("A_last10_follow_flip", last10_current),
        ("B_majority", always_maj),
        ("C_frequency", always_freq),
        ("D_markov", always_markov),
        ("E_recent", always_recent),
        ("F_pattern", always_pattern),
        ("G_ml", ml_only),
        ("H_ensemble_blend", ensemble_blend_no_override),
        ("H_ensemble_selective_m08", lambda h: ensemble_selective(h, 0.08)),
        ("H_ensemble_selective_m12", lambda h: ensemble_selective(h, 0.12)),
        ("follow", follow),
        ("flip", flip),
        ("LIVE_majority_w8", live_majority_w8_path),
    ]

    rows = [eval_method(n, rounds, fn) for n, fn in methods]
    rows.sort(key=lambda r: (r["bs_accuracy"], r["bs_n"]), reverse=True)

    print("\n=== Walk-forward Big/Small (chronological, no future) ===")
    for r in rows:
        print(
            f"{r['name']:28} BS={r['bs_accuracy']:5.2f}% "
            f"(wilsonL={r['bs_wilson_low']:5.2f}%) "
            f"n={r['bs_n']:4} skip={r['skipped']:4} "
            f"cov={r['coverage_pct']:5.1f}% "
            f"num={r['number_accuracy']:5.2f}% "
            f"col={r['color_accuracy']:5.2f}% "
            f"lift={r['lift_vs_50']:+.2f}"
        )

    best = rows[0]
    print(f"\nBEST_BY_POINT={best['name']} @{best['bs_accuracy']}% n={best['bs_n']}")
    # Prefer methods whose Wilson lower bound is best among n>=100
    eligible = [r for r in rows if r["bs_n"] >= 100]
    best_safe = max(eligible, key=lambda r: r["bs_wilson_low"]) if eligible else best
    print(
        f"BEST_WILSON_n>=100={best_safe['name']} "
        f"@{best_safe['bs_accuracy']}% wilsonL={best_safe['bs_wilson_low']}% "
        f"n={best_safe['bs_n']}"
    )

    out_path = ROOT / "exports" / "walkforward_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rounds": len(rounds),
        "min_history": 80,
        "methods": rows,
        "best_point": best,
        "best_wilson": best_safe,
        "note": (
            "Chronological walk-forward only. No labels/results altered. "
            "60-70% only claimed if supported by these numbers."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
