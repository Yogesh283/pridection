"""Compare indialotteryapi predictions vs dearapi.tashanwin actual results."""

from __future__ import annotations

import time

import requests

DEAR = "https://dearapi.tashanwin.fit/wingo30"
PRED = "https://indialotteryapi.com/wp-json/wingo/v1"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
ROUNDS = 10


def dear() -> dict:
    r = requests.get(DEAR, timeout=15, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def predict_next() -> dict:
    r = requests.get(f"{PRED}/predict?market=0.5", timeout=15, headers=HEADERS)
    r.raise_for_status()
    return r.json()["items"][0]


def next_info() -> dict:
    r = requests.get(f"{PRED}/next?market=0.5", timeout=15, headers=HEADERS)
    r.raise_for_status()
    return r.json()


def primary_color(color: str | None) -> str | None:
    if not color:
        return None
    c = str(color).lower()
    if "green" in c:
        return "GREEN"
    if "red" in c:
        return "RED"
    if "violet" in c or "purple" in c:
        return "VIOLET"
    return str(color).upper()


def big_small(n: int) -> str:
    return "BIG" if int(n) >= 5 else "SMALL"


def main() -> None:
    print(f"Comparing API2 predict vs dearapi actual for {ROUNDS} rounds...")
    print("Period IDs differ; comparing by next-settle timing.")
    print()

    bs_ok = col_ok = num_ok = total = 0
    cur = dear()
    last = str(cur.get("issueNumber"))
    print(
        "Synced dear period",
        last,
        "number",
        cur.get("number"),
        "colour",
        cur.get("colour"),
    )
    print()

    for i in range(ROUNDS):
        pred = predict_next()
        nxt = next_info()
        pred_digit = int(pred["digit"])
        pred_bs = str(pred["bigSmall"]).upper()
        pred_color = str(pred["color"]).upper()
        conf = pred.get("conf")
        remain = nxt.get("remain")
        print(
            f"[{i + 1}/{ROUNDS}] API2 -> period={pred['period']} "
            f"digit={pred_digit} {pred_bs} {pred_color} conf={conf} remain={remain}"
        )

        deadline = time.time() + 55
        actual = None
        while time.time() < deadline:
            data = dear()
            period = str(data.get("issueNumber"))
            if period != last:
                actual = data
                last = period
                break
            time.sleep(2)

        if actual is None:
            print("  TIMEOUT waiting for dearapi settle")
            continue

        actual_n = int(actual["number"])
        actual_c = primary_color(actual.get("colour"))
        actual_bs = big_small(actual_n)

        if pred_color == "VIOLET":
            color_hit = ("violet" in str(actual.get("colour", "")).lower()) or actual_n in (
                0,
                5,
            )
        else:
            color_hit = actual_c == pred_color

        num_hit = pred_digit == actual_n
        bs_hit = pred_bs == actual_bs
        total += 1
        num_ok += int(num_hit)
        bs_ok += int(bs_hit)
        col_ok += int(color_hit)

        def mark(ok: bool) -> str:
            return "OK" if ok else "X"

        print(
            f"  ACTUAL dear={last} number={actual_n} {actual_bs} "
            f"{actual.get('colour')} | "
            f"BS:{mark(bs_hit)} Color:{mark(color_hit)} Num:{mark(num_hit)}"
        )

    print()
    print("========== API2 ACCURACY vs dearapi ==========")
    print("Samples     :", total)
    if total:
        print(f"Big/Small   : {100 * bs_ok / total:.2f}%  ({bs_ok}/{total})")
        print(f"Color       : {100 * col_ok / total:.2f}%  ({col_ok}/{total})")
        print(f"Number      : {100 * num_ok / total:.2f}%  ({num_ok}/{total})")
    print("==============================================")
    print("Note: API2 conf values are NOT proven real accuracy.")


if __name__ == "__main__":
    main()
