"""Dry-run / apply repair for unresolved predictions that already have rounds.

Usage:
  python -m tools.repair_resolve_pending          # dry-run
  python -m tools.repair_resolve_pending --apply  # resolve matching rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.database import Database


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually resolve matching predictions (default is dry-run)",
    )
    args = parser.parse_args()
    db = Database()
    pending = db.get_unresolved_predictions()
    would_resolve = []
    orphans = []
    for pred in pending:
        actual = db.get_round_by_period(str(pred["target_period"]))
        if actual:
            would_resolve.append(
                {
                    "id": pred["id"],
                    "target_period": pred["target_period"],
                    "predicted_big_small": pred.get("predicted_big_small"),
                    "actual_number": actual["number"],
                    "actual_color": actual.get("color"),
                }
            )
        else:
            orphans.append(
                {
                    "id": pred["id"],
                    "target_period": pred["target_period"],
                    "note": "no matching round — leave unresolved",
                }
            )

    report = {
        "mode": "apply" if args.apply else "dry-run",
        "pending": len(pending),
        "would_resolve": len(would_resolve),
        "orphans_left": len(orphans),
        "resolve_rows": would_resolve,
        "orphan_rows": orphans,
    }
    print(json.dumps(report, indent=2, default=str))

    if args.apply:
        summary = db.resolve_pending_against_rounds()
        print(
            json.dumps(
                {
                    "applied": True,
                    "resolved_now": summary.get("resolved_now"),
                    "orphans_left": summary.get("orphans_left"),
                },
                indent=2,
            )
        )
    else:
        print("Dry-run only. Re-run with --apply to resolve matching rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
