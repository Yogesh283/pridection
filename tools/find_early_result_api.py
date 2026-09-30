"""Find which WinGo APIs publish results earlier than dearapi (~6s target)."""

from __future__ import annotations

import time
from datetime import datetime

import requests

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
DEAR = "https://dearapi.tashanwin.fit/wingo30"
ROUNDS = 6


def now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def dear_pick() -> tuple[str | None, str | None]:
    j = requests.get(DEAR, timeout=10, headers=HEADERS).json()
    p = j.get("issueNumber")
    n = j.get("number")
    return (str(p) if p is not None else None, str(n) if n is not None else None)


def hist_pick(url: str) -> tuple[str | None, str | None]:
    j = requests.get(url, timeout=10, headers=HEADERS).json()
    lst = (j.get("data") or {}).get("list") or []
    if not lst:
        return None, None
    row = lst[0]
    p, n = row.get("issueNumber"), row.get("number")
    return (str(p) if p is not None else None, str(n) if n is not None else None)


def main() -> None:
    endpoints = {
        "hist_30s": "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json",
        "hist_1m": "https://draw.ar-lottery01.com/WinGo/WinGo_1M/GetHistoryIssuePage.json",
        "hist_3m": "https://draw.ar-lottery01.com/WinGo/WinGo_3M/GetHistoryIssuePage.json",
    }

    print("Smoke check:")
    alive = {}
    for name, base in endpoints.items():
        url = f"{base}?ts={int(time.time() * 1000)}"
        try:
            r = requests.get(url, timeout=10, headers=HEADERS)
            ok = r.status_code == 200 and r.text.strip()[:1] == "{"
            print(f"  {name}: status={r.status_code} json={ok}")
            if ok:
                alive[name] = base
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: FAIL {exc}")

    # Focus: 30s hist vs dear (same game). Others only if same period namespace.
    print(f"\nWatching {ROUNDS} settles vs dearapi (poll 0.4s)...")
    print("Negative lag => that API earlier than dearapi\n")

    first: dict[tuple[str, str], tuple[str | None, float]] = {}
    events: list[dict] = []
    t0 = time.time()

    while len(events) < ROUNDS and time.time() - t0 < 240:
        snaps: dict[str, tuple[str | None, str | None, float]] = {}
        try:
            p, n = dear_pick()
            snaps["dear"] = (p, n, time.time())
        except Exception:
            pass
        for name, base in alive.items():
            if name != "hist_30s":
                continue  # only same 30s market for lag vs dear
            try:
                p, n = hist_pick(f"{base}?ts={int(time.time() * 1000)}")
                snaps[name] = (p, n, time.time())
            except Exception:
                pass

        for src, (p, n, t) in snaps.items():
            if not p:
                continue
            key = (src, p)
            if key in first:
                continue
            first[key] = (n, t)

            if src == "hist_30s" and ("dear", p) in first:
                lag = first[("hist_30s", p)][1] - first[("dear", p)][1]
                who = (
                    "HIST ~earlier"
                    if lag < -0.3
                    else ("DEAR earlier" if lag > 0.3 else "SAME")
                )
                events.append(
                    {"period": p, "number": n, "lag": lag, "who": who}
                )
                print(
                    f"{now()} {p} n={n} lag(hist-dear)={lag:+.2f}s | {who}"
                )
            elif src == "dear" and ("hist_30s", p) in first:
                if any(e["period"] == p for e in events):
                    continue
                lag = first[("hist_30s", p)][1] - first[("dear", p)][1]
                who = (
                    "HIST ~earlier"
                    if lag < -0.3
                    else ("DEAR earlier" if lag > 0.3 else "SAME")
                )
                events.append(
                    {"period": p, "number": n, "lag": lag, "who": who}
                )
                print(
                    f"{now()} {p} n={n} lag(hist-dear)={lag:+.2f}s | {who}"
                )

        time.sleep(0.4)

    print("\n========== SUMMARY ==========")
    if not events:
        print("No matched settles.")
        return
    lags = [e["lag"] for e in events]
    avg = sum(lags) / len(lags)
    near6 = sum(1 for L in lags if -8.0 <= L <= -4.0)
    print(f"Matched: {len(events)}")
    print(f"Avg lag hist-dear: {avg:+.2f}s")
    print(f"Min/Max: {min(lags):+.2f} / {max(lags):+.2f}")
    print(f"About 4-8s earlier (near 6s): {near6}/{len(events)}")
    print(
        "API with earlier results: "
        "https://draw.ar-lottery01.com/WinGo/WinGo_30S/GetHistoryIssuePage.json"
    )
    print(
        "Baseline (often later): https://dearapi.tashanwin.fit/wingo30"
    )
    if abs(avg + 6) < 2:
        print("Avg is close to ~6s earlier.")
    elif avg > -1:
        print("Avg earlier gap is SMALL (<1s) — not stably 6s.")
    else:
        print(f"Typical earlier gap ~{abs(avg):.1f}s (not fixed 6s).")


if __name__ == "__main__":
    main()
