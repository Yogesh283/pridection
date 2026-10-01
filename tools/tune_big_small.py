"""Tune Big/Small: find selective rules that beat ~50% on walk-forward."""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label

HIST = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
H = {"User-Agent": "Mozilla/5.0"}


def load_from_db() -> list[dict]:
    try:
        from data.database import Database

        rounds = Database().get_rounds()
        if len(rounds) >= 80:
            return rounds
    except Exception as exc:  # noqa: BLE001
        print("db note:", exc)
    rows = (
        requests.get(f"{HIST}?ts={int(time.time()*1000)}", timeout=20, headers=H)
        .json()
        .get("data")
        or {}
    ).get("list") or []
    out = []
    for r in reversed(rows):
        out.append(
            {
                "period": str(r["issueNumber"]),
                "number": int(r["number"]),
                "color": str(r.get("color") or ""),
            }
        )
    return out


def maj(history: list[dict], w: int = 8) -> tuple[str, int, int]:
    seq = [big_small_label(int(r["number"])) for r in history[-w:]]
    c = Counter(seq)
    big, small = c.get("BIG", 0), c.get("SMALL", 0)
    if big == small:
        # flip last on tie
        last = seq[-1]
        return ("SMALL" if last == "BIG" else "BIG"), big, small
    return ("BIG" if big > small else "SMALL"), big, small


def streak(history: list[dict]) -> tuple[str, int]:
    seq = [big_small_label(int(r["number"])) for r in history]
    last = seq[-1]
    n = 0
    for x in reversed(seq):
        if x == last:
            n += 1
        else:
            break
    return last, n


def eval_rule(name: str, fn, rounds: list[dict], start: int = 40) -> dict:
    ok = tot = skipped = 0
    for i in range(start, len(rounds)):
        tip = fn(rounds[:i])
        if tip is None:
            skipped += 1
            continue
        actual = big_small_label(int(rounds[i]["number"]))
        tot += 1
        ok += int(tip == actual)
    pct = (100.0 * ok / tot) if tot else 0.0
    return {"name": name, "n": tot, "skip": skipped, "pct": round(pct, 2), "ok": ok}


def main() -> int:
    rounds = load_from_db()
    print(f"rounds={len(rounds)}")
    if len(rounds) < 50:
        print("need more rounds")
        return 1

    def always_maj(h):
        t, _, _ = maj(h, 8)
        return t

    def always_follow(h):
        return big_small_label(int(h[-1]["number"]))

    def always_flip(h):
        last = big_small_label(int(h[-1]["number"]))
        return "BIG" if last == "SMALL" else "SMALL"

    def anti_streak3(h):
        last, n = streak(h)
        if n >= 3:
            return "BIG" if last == "SMALL" else "SMALL"
        t, _, _ = maj(h, 8)
        return t

    def strong_maj_only(h, need=2, w=8):
        t, big, small = maj(h, w)
        if abs(big - small) < need:
            return None  # skip weak
        return t

    def strong_maj_or_streak(h):
        last, n = streak(h)
        if n >= 3:
            return "BIG" if last == "SMALL" else "SMALL"
        return strong_maj_only(h, need=2, w=8)

    def strong_maj3(h):
        return strong_maj_only(h, need=3, w=10)

    rules = [
        ("maj8", always_maj),
        ("follow", always_follow),
        ("flip", always_flip),
        ("anti_streak3+maj", anti_streak3),
        ("strong_maj_margin2", lambda h: strong_maj_only(h, 2, 8)),
        ("strong_maj_margin3", lambda h: strong_maj_only(h, 3, 10)),
        ("streak3_or_strong", strong_maj_or_streak),
        ("strong_maj3", strong_maj3),
    ]
    rows = [eval_rule(n, f, rounds) for n, f in rules]
    rows.sort(key=lambda r: (r["pct"], r["n"]), reverse=True)
    for r in rows:
        print(
            f"{r['name']:22} pct={r['pct']:5.2f}%  n={r['n']:4}  "
            f"ok={r['ok']:4}  skipped={r['skip']:4}"
        )
    best = rows[0]
    print(f"BEST={best['name']} @{best['pct']}% (tips={best['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
