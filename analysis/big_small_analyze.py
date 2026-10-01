"""Analyze past Big/Small rounds, then derive the next tip from that analysis."""

from __future__ import annotations

from collections import Counter
from typing import Any

from analysis.statistics import big_small_label

# WinGo 30s: 30 minutes ≈ 60 settled rounds.
ANALYSIS_ROUNDS_30M = 60
# Need a clearer majority on a long window (not just 1–2 vote edge).
MIN_MARGIN_30M = 6


def analyze_big_small(
    history: list[dict[str, Any]], window: int = ANALYSIS_ROUNDS_30M
) -> dict[str, Any]:
    """
    Step 1 — analyze last ~30 minutes of settled history (default 60 rounds).
    Step 2 — caller uses this dict to choose TIP / WAIT.
    """
    if not history:
        return {
            "ok": False,
            "reason": "no_history",
            "labels": [],
        }

    labels = [big_small_label(int(r["number"])) for r in history]
    last = labels[-1]
    streak_n = 0
    for x in reversed(labels):
        if x == last:
            streak_n += 1
        else:
            break

    win = labels[-window:] if len(labels) >= window else labels
    counts = Counter(win)
    big_n = int(counts.get("BIG", 0))
    small_n = int(counts.get("SMALL", 0))
    margin = abs(big_n - small_n)
    last20 = labels[-20:] if len(labels) >= 20 else labels
    c20 = Counter(last20)

    # Recent rule hit-rates on the same ~30m window
    follow_hits = flip_hits = maj_hits = tested = 0
    start = max(8, len(labels) - window)
    for i in range(start, len(labels)):
        past = labels[:i]
        actual = labels[i]
        follow = past[-1]
        flip = "BIG" if follow == "SMALL" else "SMALL"
        w = past[-window:] if len(past) >= window else past
        cw = Counter(w)
        if cw.get("BIG", 0) == cw.get("SMALL", 0):
            maj = flip
        else:
            maj = "BIG" if cw.get("BIG", 0) > cw.get("SMALL", 0) else "SMALL"
        follow_hits += int(follow == actual)
        flip_hits += int(flip == actual)
        maj_hits += int(maj == actual)
        tested += 1

    def rate(h: int) -> float:
        return round(100.0 * h / tested, 2) if tested else 50.0

    return {
        "ok": True,
        "history_count": len(labels),
        "analysis_minutes": round(len(win) * 0.5, 1),  # 30s rounds
        "analysis_rounds": len(win),
        "last_bs": last,
        "streak": streak_n,
        "window": len(win),
        "window_big": big_n,
        "window_small": small_n,
        "window_margin": margin,
        "min_margin": MIN_MARGIN_30M,
        "last20_big": int(c20.get("BIG", 0)),
        "last20_small": int(c20.get("SMALL", 0)),
        "recent_test_n": tested,
        "recent_follow_pct": rate(follow_hits),
        "recent_flip_pct": rate(flip_hits),
        "recent_majority_pct": rate(maj_hits),
        "last5": labels[-5:],
    }


def tip_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Step 2 — prediction ONLY from analysis fields above.
    """
    if not analysis.get("ok"):
        return {
            "big_small": None,
            "skip": True,
            "action": "WAIT",
            "source": "no_analysis",
            "probs": {"BIG": 0.5, "SMALL": 0.5},
            "reason": analysis.get("reason") or "missing",
        }

    last = str(analysis["last_bs"])
    streak_n = int(analysis["streak"])
    big_n = int(analysis["window_big"])
    small_n = int(analysis["window_small"])
    margin = int(analysis["window_margin"])
    total_w = max(1, int(analysis["window"]))
    need = int(analysis.get("min_margin") or MIN_MARGIN_30M)
    minutes = analysis.get("analysis_minutes") or round(total_w * 0.5, 1)

    # Rule A: analyze streak -> if 3+ same, tip opposite
    if streak_n >= 3:
        tip = "SMALL" if last == "BIG" else "BIG"
        other = last
        return {
            "big_small": tip,
            "skip": False,
            "action": "TIP",
            "source": "analysis_anti_streak3",
            "probs": {tip: 0.58, other: 0.42},
            "reason": f"last {streak_n}x {last} -> tip {tip} (from ~{minutes}m data)",
        }

    # Rule B: ~30m majority window needs clear margin
    if big_n == small_n or margin < need:
        return {
            "big_small": None,
            "skip": True,
            "action": "WAIT",
            "source": "analysis_weak_signal",
            "probs": {"BIG": 0.5, "SMALL": 0.5},
            "reason": (
                f"~{minutes}m ({total_w}r): BIG={big_n} SMALL={small_n} "
                f"margin={margin} need>={need} -> WAIT"
            ),
        }

    tip = "BIG" if big_n > small_n else "SMALL"
    probs = {
        "BIG": (big_n + 1) / (total_w + 2),
        "SMALL": (small_n + 1) / (total_w + 2),
    }
    s = probs["BIG"] + probs["SMALL"]
    probs = {k: v / s for k, v in probs.items()}
    return {
        "big_small": tip,
        "skip": False,
        "action": "TIP",
        "source": "analysis_30m_majority",
        "probs": probs,
        "reason": (
            f"~{minutes}m ({total_w}r): BIG={big_n} SMALL={small_n} "
            f"margin={margin} -> tip {tip}"
        ),
    }


def analyze_then_predict(history: list[dict[str, Any]]) -> dict[str, Any]:
    analysis = analyze_big_small(history, window=ANALYSIS_ROUNDS_30M)
    tip = tip_from_analysis(analysis)
    return {"analysis": analysis, "tip": tip}
