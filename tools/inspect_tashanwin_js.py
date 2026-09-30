"""Fetch tashanwin.com frontend JS and search for draw/API/predict logic."""

from __future__ import annotations

import re
from pathlib import Path

import requests

H = {"User-Agent": "Mozilla/5.0"}
BASE = "https://tashanwin.com"
OUT = Path(__file__).resolve().parents[1] / "exports" / "tashanwin_js_findings.txt"


def main() -> int:
    lines: list[str] = []
    index = requests.get(f"{BASE}/assets/js/index.js?id=333", timeout=30, headers=H)
    text = index.text
    lines.append(f"index.js status={index.status_code} bytes={len(text)}")

    patterns = [
        r"draw\.ar-lottery",
        r"GetHistoryIssuePage",
        r"GetNoaverageEmerdList",
        r"dearapi",
        r"wingo30",
        r"WinGo_30S",
        r"predict",
        r"Missing",
        r"issueNumber",
        r"bigSmall|BigSmall|big_small",
        r"https?://[a-zA-Z0-9._/-]+",
    ]
    for pat in patterns:
        hits = re.findall(pat, text, flags=re.I)
        uniq = list(dict.fromkeys(hits))[:20]
        lines.append(f"PAT {pat!r}: count={len(hits)} sample={uniq[:8]}")

    # Extract interesting string literals containing api/lottery/wingo
    urls = re.findall(r"[\"'](https?://[^\"']+)[\"']", text)
    interesting = [
        u
        for u in dict.fromkeys(urls)
        if any(k in u.lower() for k in ("api", "draw", "lottery", "wingo", "game", "cdn"))
    ]
    lines.append("URLS:")
    for u in interesting[:40]:
        lines.append(f"  {u}")

    # Also relative API paths
    rel = re.findall(r"[\"'](/[^\"']*(?:api|wingo|lottery|history|chart)[^\"']*)[\"']", text, re.I)
    lines.append("REL_PATHS:")
    for u in list(dict.fromkeys(rel))[:40]:
        lines.append(f"  {u}")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:80]))
    print(f"\nSaved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
