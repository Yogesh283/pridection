"""Analyze past Big/Small + number rounds, then derive the next tip."""

from __future__ import annotations

from collections import Counter
from typing import Any

from analysis.statistics import big_small_label

# Tip from last 10 settled rounds only (~5 minutes on WinGo 30s).
ANALYSIS_ROUNDS = 10
# Need clear recent edge before TIP (else WAIT).
MIN_RULE_PCT = 60.0
MIN_RULE_EDGE = 20.0


def analyze_big_small(
    history: list[dict[str, Any]], window: int = ANALYSIS_ROUNDS
) -> dict[str, Any]:
    """
    Step 1 — analyze last N settled rounds (default 10).
    Step 2 — caller uses this dict to choose TIP / WAIT + number.
    """
    if not history:
        return {
            "ok": False,
            "reason": "no_history",
            "labels": [],
        }

    labels = [big_small_label(int(r["number"])) for r in history]
    numbers = [int(r["number"]) for r in history]
    last = labels[-1]
    last_num = numbers[-1]
    streak_n = 0
    for x in reversed(labels):
        if x == last:
            streak_n += 1
        else:
            break

    win = labels[-window:] if len(labels) >= window else labels
    win_nums = numbers[-window:] if len(numbers) >= window else numbers
    counts = Counter(win)
    big_n = int(counts.get("BIG", 0))
    small_n = int(counts.get("SMALL", 0))
    margin = abs(big_n - small_n)

    num_counts = Counter(win_nums)
    top_nums = [
        {"number": int(n), "count": int(c), "pct": round(100.0 * c / len(win_nums), 1)}
        for n, c in num_counts.most_common(5)
    ]

    # follow = same as last, flip = opposite — measure which hit on last 10.
    follow_hits = flip_hits = tested = 0
    start = max(2, len(labels) - window)
    for i in range(start, len(labels)):
        past = labels[:i]
        actual = labels[i]
        follow = past[-1]
        flip = "BIG" if follow == "SMALL" else "SMALL"
        follow_hits += int(follow == actual)
        flip_hits += int(flip == actual)
        tested += 1

    def rate(h: int) -> float:
        return round(100.0 * h / tested, 2) if tested else 50.0

    return {
        "ok": True,
        "history_count": len(labels),
        "analysis_minutes": round(len(win) * 0.5, 1),
        "analysis_rounds": len(win),
        "last_bs": last,
        "last_number": last_num,
        "streak": streak_n,
        "window": len(win),
        "window_big": big_n,
        "window_small": small_n,
        "window_margin": margin,
        "min_rule_pct": MIN_RULE_PCT,
        "min_rule_edge": MIN_RULE_EDGE,
        "recent_test_n": tested,
        "recent_follow_pct": rate(follow_hits),
        "recent_flip_pct": rate(flip_hits),
        "last10": win,
        "last10_numbers": win_nums,
        "number_top": top_nums,
        "last5": labels[-5:],
    }


def _pick_number(analysis: dict[str, Any], side: str | None) -> dict[str, Any]:
    """Most frequent digit in last 10; if BS side given, prefer that side."""
    win_nums = list(analysis.get("last10_numbers") or [])
    if not win_nums:
        return {"number": None, "alts": [], "source": "none"}

    def rank(pool: list[int]) -> list[dict[str, Any]]:
        c = Counter(pool)
        total = len(pool) or 1
        return [
            {
                "number": int(n),
                "count": int(cnt),
                "pct": round(100.0 * cnt / total, 1),
            }
            for n, cnt in c.most_common(3)
        ]

    if side in ("BIG", "SMALL"):
        side_pool = [n for n in win_nums if big_small_label(n) == side]
        if side_pool:
            ranked = rank(side_pool)
            return {
                "number": ranked[0]["number"],
                "alts": ranked,
                "source": f"last10_freq_{side.lower()}",
                "pct": ranked[0]["pct"],
            }

    ranked = rank(win_nums)
    return {
        "number": ranked[0]["number"],
        "alts": ranked,
        "source": "last10_freq",
        "pct": ranked[0]["pct"],
    }


def tip_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Step 2 — tip BS when last-10 clearly favors follow OR flip.
    Always attach a last-10 frequency number tip.
    """
    if not analysis.get("ok"):
        return {
            "big_small": None,
            "number": None,
            "numbers": [],
            "skip": True,
            "action": "WAIT",
            "source": "no_analysis",
            "probs": {"BIG": 0.5, "SMALL": 0.5},
            "reason": analysis.get("reason") or "missing",
        }

    last = str(analysis["last_bs"])
    last10 = analysis.get("last10") or []
    follow_pct = float(analysis.get("recent_follow_pct") or 50.0)
    flip_pct = float(analysis.get("recent_flip_pct") or 50.0)
    need_pct = float(analysis.get("min_rule_pct") or MIN_RULE_PCT)
    need_edge = float(analysis.get("min_rule_edge") or MIN_RULE_EDGE)

    follow_tip = last
    flip_tip = "BIG" if last == "SMALL" else "SMALL"

    if follow_pct >= flip_pct:
        best_name, best_tip, best_pct = "follow", follow_tip, follow_pct
        other_pct = flip_pct
    else:
        best_name, best_tip, best_pct = "flip", flip_tip, flip_pct
        other_pct = follow_pct
    edge = best_pct - other_pct

    if best_pct < need_pct or edge < need_edge:
        num = _pick_number(analysis, None)
        return {
            "big_small": None,
            "number": num.get("number"),
            "numbers": num.get("alts") or [],
            "number_source": num.get("source"),
            "number_pct": num.get("pct"),
            "skip": True,
            "action": "WAIT",
            "source": "last10_no_edge",
            "probs": {"BIG": 0.5, "SMALL": 0.5},
            "reason": (
                f"last10 {last10}: follow={follow_pct}% flip={flip_pct}% "
                f"best={best_name}@{best_pct}% edge={edge:.1f} "
                f"need>={need_pct}%/+{need_edge} -> WAIT"
            ),
        }

    other = "BIG" if best_tip == "SMALL" else "SMALL"
    conf = min(0.70, 0.50 + (best_pct - 50.0) / 100.0)
    num = _pick_number(analysis, best_tip)
    return {
        "big_small": best_tip,
        "number": num.get("number"),
        "numbers": num.get("alts") or [],
        "number_source": num.get("source"),
        "number_pct": num.get("pct"),
        "skip": False,
        "action": "TIP",
        "source": f"last10_{best_name}",
        "probs": {best_tip: conf, other: 1.0 - conf},
        "reason": (
            f"last10 {last10}: {best_name}@{best_pct}% > "
            f"{'flip' if best_name == 'follow' else 'follow'}@{other_pct}% "
            f"-> tip {best_tip} num={num.get('number')}"
        ),
    }


def analyze_then_predict(history: list[dict[str, Any]]) -> dict[str, Any]:
    analysis = analyze_big_small(history, window=ANALYSIS_ROUNDS)
    tip = tip_from_analysis(analysis)
    return {"analysis": analysis, "tip": tip}
