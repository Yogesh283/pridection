"""Compare settle timing between dearapi and draw.ar-lottery01 WinGo_30S."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

DEAR = "https://dearapi.tashanwin.fit/wingo30"
HIST = "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
ROUNDS = 8


def now_ms() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def fetch_dear() -> tuple[str | None, str | None, str | None]:
    r = requests.get(DEAR, timeout=12, headers=HEADERS)
    r.raise_for_status()
    j = r.json()
    return (
        str(j.get("issueNumber")) if j.get("issueNumber") is not None else None,
        str(j.get("number")) if j.get("number") is not None else None,
        str(j.get("colour") or j.get("color") or ""),
    )


def fetch_hist() -> tuple[str | None, str | None, str | None]:
    url = f"{HIST}?ts={int(time.time() * 1000)}"
    r = requests.get(url, timeout=12, headers=HEADERS)
    r.raise_for_status()
    j = r.json()
    lst = (j.get("data") or {}).get("list") or []
    if not lst:
        return None, None, None
    row = lst[0]
    return (
        str(row.get("issueNumber")) if row.get("issueNumber") is not None else None,
        str(row.get("number")) if row.get("number") is not None else None,
        str(row.get("color") or row.get("colour") or ""),
    )


def main() -> None:
    print("Timing sync test: dearapi vs draw.ar-lottery01 WinGo_30S")
    print(f"Watching {ROUNDS} new settles (poll every 1s)...")
    print()

    dear_p, dear_n, dear_c = fetch_dear()
    hist_p, hist_n, hist_c = fetch_hist()
    print(f"START dear={dear_p} n={dear_n} | hist={hist_p} n={hist_n} | same={dear_p==hist_p}")
    print()

    seen = set()
    if dear_p:
        seen.add(dear_p)
    events = []
    last_dear = dear_p
    last_hist = hist_p
    pending_dear = None  # (period, number, color, t)
    pending_hist = None

    deadline = time.time() + ROUNDS * 45 + 60
    while len(events) < ROUNDS and time.time() < deadline:
        t0 = time.monotonic()
        try:
            d_p, d_n, d_c = fetch_dear()
        except Exception as exc:
            print(now_ms(), "dear FAIL", exc)
            d_p = last_dear
            d_n = d_c = None
        try:
            h_p, h_n, h_c = fetch_hist()
        except Exception as exc:
            print(now_ms(), "hist FAIL", exc)
            h_p = last_hist
            h_n = h_c = None

        wall = time.time()

        if d_p and d_p != last_dear:
            pending_dear = (d_p, d_n, d_c, wall)
            print(f"{now_ms()} DEAR new  {d_p} number={d_n} {d_c}")
            last_dear = d_p

        if h_p and h_p != last_hist:
            pending_hist = (h_p, h_n, h_c, wall)
            print(f"{now_ms()} HIST new  {h_p} number={h_n} {h_c}")
            last_hist = h_p

        # When both have same new period observed, record lag.
        if pending_dear and pending_hist and pending_dear[0] == pending_hist[0]:
            period = pending_dear[0]
            if period not in seen:
                lag = pending_hist[3] - pending_dear[3]
                same_num = pending_dear[1] == pending_hist[1]
                events.append(
                    {
                        "period": period,
                        "lag_hist_minus_dear_sec": lag,
                        "same_number": same_num,
                        "dear_number": pending_dear[1],
                        "hist_number": pending_hist[1],
                    }
                )
                who = (
                    "SAME time (~0)"
                    if abs(lag) < 0.5
                    else ("HIST later" if lag > 0 else "DEAR later")
                )
                print(
                    f"{now_ms()} MATCH {period} | "
                    f"lag(hist-dear)={lag:+.2f}s | number_same={same_num} | {who}"
                )
                print()
                seen.add(period)
                pending_dear = None
                pending_hist = None

        # if different periods pending too long, keep waiting
        elapsed = time.monotonic() - t0
        time.sleep(max(0.2, 1.0 - elapsed))

    print("========== TIMING SUMMARY ==========")
    print("Matched settles:", len(events))
    if events:
        lags = [e["lag_hist_minus_dear_sec"] for e in events]
        same = sum(1 for e in events if e["same_number"])
        print(f"Number match     : {same}/{len(events)}")
        print(f"Avg lag hist-dear: {sum(lags)/len(lags):+.2f}s")
        print(f"Min lag          : {min(lags):+.2f}s")
        print(f"Max lag          : {max(lags):+.2f}s")
        print("Positive lag => history API updated AFTER dearapi")
        print("Negative lag => history API updated BEFORE dearapi")
        for e in events:
            print(
                f"  {e['period']}: lag={e['lag_hist_minus_dear_sec']:+.2f}s "
                f"num {e['dear_number']}/{e['hist_number']}"
            )
    print("====================================")


if __name__ == "__main__":
    main()
