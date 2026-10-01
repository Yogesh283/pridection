"""Audit real Big/Small + Color accuracy on recent WinGo history."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics import big_small_label
from api.client import primary_color
from models.ensemble import EnsemblePredictor
from models.predictor import MajorityWindowModel

HIST = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
H = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def load_rounds() -> list[dict]:
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


def eval_model(name: str, predict_fn, rounds: list[dict], start: int = 5) -> dict:
    bs_ok = col_ok = tot = 0
    for i in range(start, len(rounds)):
        out = predict_fn(rounds[:i])
        if not out:
            continue
        act = rounds[i]
        ab = big_small_label(int(act["number"]))
        ac = primary_color(act.get("color"))
        if "top_big_small" in out:
            pb = str(out["top_big_small"])
            pc = str(out.get("top_color") or "")
        else:
            pb = max(out["big_small"], key=out["big_small"].get)
            pc = max(out["colors"], key=out["colors"].get)
        tot += 1
        bs_ok += int(pb == ab)
        col_ok += int(bool(ac) and pc == ac)
    return {
        "name": name,
        "n": tot,
        "bs": round(100.0 * bs_ok / tot, 2) if tot else 0.0,
        "color": round(100.0 * col_ok / tot, 2) if tot else 0.0,
    }


def main() -> int:
    rounds = load_rounds()
    print(f"history rows={len(rounds)} latest={rounds[-1] if rounds else None}")
    maj = MajorityWindowModel()
    ens = EnsemblePredictor()
    ens.maybe_train_ml(rounds)

    def follow(h):
        last = big_small_label(int(h[-1]["number"]))
        col = primary_color(h[-1].get("color")) or "RED"
        return {
            "top_big_small": last,
            "top_color": col,
            "big_small": {last: 0.55, "BIG" if last == "SMALL" else "SMALL": 0.45},
            "colors": {"RED": 0.5, "GREEN": 0.5, "VIOLET": 0.0},
        }

    def flip(h):
        last = big_small_label(int(h[-1]["number"]))
        nxt = "BIG" if last == "SMALL" else "SMALL"
        return {
            "top_big_small": nxt,
            "top_color": "RED" if nxt == "BIG" else "GREEN",
            "big_small": {nxt: 0.55, last: 0.45},
            "colors": {"RED": 0.5, "GREEN": 0.5, "VIOLET": 0.0},
        }

    rows = [
        eval_model("majority", maj.predict, rounds),
        eval_model(
            "ensemble",
            lambda h: ens.predict(h, train_ml=False, focus="big_small"),
            rounds,
        ),
        eval_model("follow_last", follow, rounds),
        eval_model("flip_last", flip, rounds),
    ]
    for r in rows:
        print(
            f"{r['name']:12} n={r['n']:3}  BS={r['bs']:5.1f}%  COLOR={r['color']:5.1f}%"
        )
    best = max(rows, key=lambda x: x["bs"])
    print(f"BEST_BS={best['name']} @{best['bs']}% (random~50%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
