"""
Wingo 30 statistical analyzer.

Collects publicly available results, estimates probabilities from history,
and measures real settled accuracy. Does not place bets or connect to wallets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as a script.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.accuracy import compute_accuracy_report, format_accuracy_text
from analysis.statistics import big_small_label, summarize_rounds
from api.client import WingoAPIClient, print_schema_diagnostic
from config import LOG_PATH, WINGO_API_URL
from data.collector import collect_for_hours, collect_once
from data.database import Database
from models.backtest import backtest_history, print_backtest_report
from models.ensemble import (
    EnsemblePredictor,
    build_prediction_export,
    write_prediction_json,
)
from scheduler.runner import AnalyzerRunner, run_dashboard_snapshot


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def cmd_collect() -> int:
    summary = collect_once()
    if not summary.get("ok"):
        print(f"Collect failed: {summary.get('error')}")
        return 1
    print(
        f"Collected OK. inserted={summary.get('inserted')} "
        f"current={((summary.get('current') or {}).get('period'))}"
    )
    return 0


def cmd_collect_hours(hours: float) -> int:
    summary = collect_for_hours(hours=hours)
    print(
        f"Done. inserted={summary.get('inserted')} "
        f"total_rounds_in_db={summary.get('stored_rounds')} "
        f"last_period={summary.get('last_period')}"
    )
    return 0


def cmd_inspect() -> int:
    client = WingoAPIClient()
    raw = client.fetch_raw()
    print(f"API URL: {WINGO_API_URL}")
    print_schema_diagnostic(raw)
    normalized = client.fetch_normalized()
    print("Normalized current:", normalized.get("current"))
    print("History items:", len(normalized.get("history") or []))
    return 0


def cmd_predict() -> int:
    # Quiet console during predict — only clean result matters.
    import warnings

    warnings.filterwarnings("ignore")
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.ERROR)

    # Use the same guarded, persisted path as continuous production.
    bundle = AnalyzerRunner().generate_prediction(force=True, train_ml=False)
    if not bundle:
        print("PREDICTION_UNAVAILABLE")
        return 1
    live_result = bundle["result"]
    if live_result.get("skip_tip"):
        print("PREDICTION_UNAVAILABLE")
        print(
            f"Reason: {live_result.get('unavailable_reason') or 'model_not_ready'}"
        )
        return 0
    print(f"target_period: {bundle['target_period']}")
    print(f"prediction: {live_result['top_big_small']}")
    print(f"probability_big: {live_result.get('probability_big')}")
    print(f"probability_small: {live_result.get('probability_small')}")
    print(f"confidence: {live_result.get('confidence_score')}")
    print(f"model_name: {live_result.get('model_name')}")
    print(f"model_version: {live_result.get('model_version')}")
    print("status: PREDICTION_AVAILABLE")
    return 0

    # Always refresh from API first so prediction tracks the latest settled round.
    summary = collect_once()
    db = Database()

    # Resolve any open predictions that already have actual rounds.
    pending = db.get_unresolved_predictions()
    rounds_by_period = {r["period"]: r for r in db.get_rounds()}
    for pred in pending:
        actual = rounds_by_period.get(pred["target_period"])
        if actual:
            db.resolve_prediction(
                pred["id"],
                actual_number=int(actual["number"]),
                actual_color=actual.get("color"),
            )

    from api.history_sync import guess_next_period, sync_history_into_db

    try:
        sync_history_into_db(db)
    except Exception as exc:  # noqa: BLE001
        print(f"history sync note: {exc}")

    rounds = db.get_rounds()
    if len(rounds) < 5:
        print("Data kam hai. Pehle collect chalao.")
        return 1

    latest = rounds[-1]
    current = str(latest["period"])
    target = guess_next_period(current)

    predictor = EnsemblePredictor()
    resolved = db.get_resolved_predictions()
    hist_bs = None
    hist_num = None
    if resolved:
        report = compute_accuracy_report(resolved)
        hist_bs = float(report.get("big_small_accuracy") or 0.0)
        hist_num = float(report.get("number_accuracy") or 0.0)
    result = predictor.predict(
        rounds,
        historical_big_small_accuracy=hist_bs,
        historical_number_accuracy=hist_num,
        train_ml=True,
        focus="big_small",
    )
    if not result:
        print("Prediction nahi ban payi.")
        return 1

    export = build_prediction_export(target, result, current_period=current)
    write_prediction_json(export)
    decision = str(result.get("decision_strategy") or "gradient_boosting")
    if not result.get("skip_tip") and result.get("top_big_small"):
        db.save_prediction(
            {
                "target_period": target,
                "predicted_number": result.get("top_number"),
                "predicted_color": result.get("top_color"),
                "predicted_big_small": result["top_big_small"],
                "number_probability": result.get("number_probability"),
                "color_probability": result.get("color_probability"),
                "big_small_probability": result.get("big_small_probability"),
                "model_name": decision,
                "decision_strategy": decision,
                "status": "TIP",
            },
            update_existing=True,
        )

    # Clean output only — no logs noise for the user-facing lines.
    print("")
    print("====================================")
    print("  NEXT PREDICTION (Gradient Boosting)")
    print("====================================")
    print(f"Period     : {target}")
    if result.get("skip_tip"):
        print(f"Big/Small  : {result.get('decision_strategy')}")
        if result.get("unavailable_reason"):
            print(f"Reason     : {result.get('unavailable_reason')}")
    else:
        print(f"Big/Small  : {result['top_big_small']}")
        if result.get("probability_big") is not None:
            print(
                f"Probability: BIG {float(result['probability_big']):.2f} | "
                f"SMALL {float(result['probability_small']):.2f}"
            )
    print(
        f"Confidence : {result['confidence_level']} "
        f"({result['confidence_score'] * 100:.0f}%)"
    )
    print(f"Model      : {decision}")
    if result.get("model_version"):
        print(f"Version    : {result.get('model_version')}")
    if result.get("trained_on_rows"):
        print(f"Trained on : {result.get('trained_on_rows')} rows")
    print("====================================")
    print("Note: estimate only, guarantee nahi")
    print("")
    if summary.get("ok") and not summary.get("inserted"):
        print("Same round abhi — 30 sec baad dubara try karo.")
    return 0


def cmd_backtest() -> int:
    db = Database()
    rounds = db.get_rounds()
    if len(rounds) < 40:
        print(
            f"Need more historical rounds for a meaningful backtest "
            f"(have {len(rounds)}). Keep collecting."
        )
        if len(rounds) < 10:
            return 1
    report = backtest_history(rounds, min_history=min(30, max(5, len(rounds) // 3)))
    print_backtest_report(report)
    print(f"Last 50 number accuracy: {report['last_50']['number_accuracy']:.2f}%")
    print(f"Last 20 number accuracy: {report['last_20']['number_accuracy']:.2f}%")
    print(f"Last 50 Big/Small accuracy: {report['last_50'].get('big_small_accuracy', report.get('big_small_accuracy', 0.0)):.2f}%")
    print(f"Last 20 Big/Small accuracy: {report['last_20'].get('big_small_accuracy', 0.0):.2f}%")
    db.upsert_model_metrics(
        "backtest_ensemble",
        sample_size=report["total_predictions"],
        number_accuracy=report["number_accuracy"],
        color_accuracy=report["color_accuracy"],
        big_small_accuracy=report.get("big_small_accuracy", 0.0),
    )
    db.export_csv()
    return 0


def cmd_dashboard() -> int:
    run_dashboard_snapshot()
    return 0


def cmd_stats() -> int:
    db = Database()
    rounds = db.get_rounds()
    print(f"Stored rounds: {len(rounds)}")
    if rounds:
        summary = summarize_rounds(rounds)
        print(f"Big ratio: {summary['big_ratio']:.3f}")
        print("Color frequencies:", summary["color_frequency"])
    resolved = db.get_resolved_predictions()
    report = compute_accuracy_report(resolved)
    print("--- Settled prediction accuracy ---")
    print(format_accuracy_text(report))
    metrics = db.get_model_metrics()
    if metrics:
        print("--- Stored model metrics ---")
        for row in metrics:
            print(
                f"{row['model_name']}: n={row['sample_size']} "
                f"big_small={row.get('big_small_accuracy') or 0:.2f}% "
                f"color={row['color_accuracy']:.2f}% "
                f"number={row['number_accuracy']:.2f}%"
            )
    db.export_csv()
    return 0


def cmd_run() -> int:
    # Default continuous = 1 hour live with confidence + auto accuracy.
    return cmd_live(1.0)


def cmd_live(hours: float) -> int:
    import warnings

    warnings.filterwarnings("ignore")
    # Quiet noisy logs; keep clean prediction/confidence/accuracy output.
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.setLevel(logging.ERROR)
    runner = AnalyzerRunner()
    runner.run_live(hours=hours)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wingo 30 statistical analyzer. Measures real historical accuracy. "
            "Does not place bets."
        )
    )
    parser.add_argument("--collect", action="store_true", help="Fetch and store once")
    parser.add_argument(
        "--collect-hours",
        type=float,
        metavar="HOURS",
        help="Auto-fetch for N hours and store every new round in MySQL",
    )
    parser.add_argument("--inspect", action="store_true", help="Inspect API schema")
    parser.add_argument("--predict", action="store_true", help="Generate one prediction")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--dashboard", action="store_true", help="Show terminal dashboard")
    parser.add_argument("--run", action="store_true", help="Start continuous live analyzer")
    parser.add_argument(
        "--live",
        type=float,
        metavar="HOURS",
        help="Live predict forever if 0, else N hours (example: --live 0)",
    )
    parser.add_argument("--stats", action="store_true", help="Show stats and accuracy")
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    # Ensure folders exist.
    (ROOT / "storage").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "exports").mkdir(parents=True, exist_ok=True)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.live is not None:
        return cmd_live(args.live)
    if args.collect_hours is not None:
        return cmd_collect_hours(args.collect_hours)
    if args.collect:
        return cmd_collect()
    if args.inspect:
        return cmd_inspect()
    if args.predict:
        return cmd_predict()
    if args.backtest:
        return cmd_backtest()
    if args.dashboard:
        return cmd_dashboard()
    if args.run:
        return cmd_run()
    if args.stats:
        return cmd_stats()

    parser.print_help()
    print("")
    print("LIVE prediction (confidence + accuracy):")
    print("  python -u main.py --live 1")
    print("  OR double-click: live.bat")
    print("")
    print("One prediction:")
    print("  python main.py --predict")
    print("Accuracy:")
    print("  python main.py --stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
