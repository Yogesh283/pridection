# Wingo 30 Predictor

Statistical / pattern-analysis system for publicly available **Wingo 30** game results.

This project:

- Collects public result data from an HTTP API
- Stores settled rounds in SQLite
- Builds probability estimates from historical patterns
- Records whether each prediction was correct **after** the next round settles
- Reports **real** historical accuracy (never invented)

This project does **not**:

- Claim or fake 100% accuracy
- Place bets automatically
- Connect to wallets, payments, or betting systems

Probability is **not** certainty. A random digit has a 10% base rate; color markets are also noisy. Treat every output as an estimate for analysis only.

## Observed API schema

Endpoint:

`https://dearapi.tashanwin.fit/wingo30`

Method: `GET` (no auth / body required in live inspection)

Example response:

```json
{
  "status": "ok",
  "issueNumber": "20260918100050199",
  "number": "0",
  "colour": "red,violet"
}
```

| Field | Meaning |
| --- | --- |
| `issueNumber` | Period / round ID |
| `number` | Settled digit `0-9` (string) |
| `colour` | Settled colour (`red`, `green`, or compound `red,violet` / `green,violet`) |
| `status` | Response status |

The live endpoint does **not** currently return historical lists or countdown timers. History is built by polling over time. If countdown appears later, the scheduler will synchronize to it.

The client still inspects nested JSON dynamically and will pause predictions if the schema changes and required fields disappear.

## 1. Python installation

Install Python 3.10+ from [python.org](https://www.python.org/downloads/).

## 2. Virtual environment (Windows)

```bat
cd wingo30_predictor
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bat
pip install -r requirements.txt
```

## 4. Environment configuration

```bat
copy .env.example .env
```

Edit `.env` if needed:

```env
WINGO_API_URL=https://dearapi.tashanwin.fit/wingo30
REQUEST_TIMEOUT=10
MIN_TRAINING_SAMPLES=200
POLL_INTERVAL_SECONDS=30
```

## 5. API schema inspection

```bat
python main.py --inspect
```

This prints a key/type diagnostic of the live JSON and the normalized internal record.

## 6. Database (XAMPP MySQL)

This project stores data in **XAMPP MySQL/MariaDB** (not SQLite).

1. Start **Apache + MySQL** from the XAMPP Control Panel.
2. Default `.env` settings:

```env
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=
DB_NAME=wingo30
```

3. On first run the app auto-creates database `wingo30` and tables.
   Optional manual SQL: `storage/wingo30_mysql.sql` (phpMyAdmin import).

Tables:

- `rounds` — settled results (`period` unique)
- `predictions` — forecasts + settlement outcome
- `model_metrics` — stored accuracy snapshots

Check in phpMyAdmin: `http://localhost/phpmyadmin` → database `wingo30`.

## 7. Collect historical data

One-shot:

```bat
python main.py --collect
```

Auto-fetch for 1 hour (~120 rounds, every ~30s) into MySQL:

```bat
python main.py --collect-hours 1
```

Keep the terminal open until it finishes. All unique periods are stored in `wingo30.rounds`.

## 8. Training

ML models (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`) train only when at least `MIN_TRAINING_SAMPLES` (default 200) settled rounds exist.

Training uses a **chronological** split (first 80% train / last 20% test). Time-series data is never randomly shuffled.

## 9. Backtesting

```bat
python main.py --backtest
```

For every historical round, the system:

1. Uses **only** earlier rounds
2. Generates a prediction
3. Compares with the actual settled result
4. Aggregates number / color / big-small accuracy

## 10. Live analyzer

```bat
python main.py --run
```

Loop (~30 seconds, smart sleep):

1. Fetch API
2. Detect new settled period
3. Resolve previous prediction
4. Generate next prediction
5. Write `prediction.json` + CSV exports
6. Sleep using countdown when available, otherwise poll interval

## 11. Interpreting accuracy

Accuracy is computed only from **settled** predictions.

- If accuracy is 51%, the dashboard shows 51%.
- If accuracy is 43%, it shows 43%.
- With fewer than 20 settled samples, the UI shows **Insufficient sample size.**

Never treat a high short-window score as proof the next result is known.

## 12. Limitations

- Wingo-style draws are effectively high-entropy; patterns can be illusory.
- The public API currently exposes only the latest result, so cold starts need time.
- Compound colours (e.g. `red,violet`) are normalized; primary colour metrics prefer RED/GREEN.
- Ensemble probabilities are scores, not guarantees.
- This software is for statistical analysis / education only.

## CLI commands

```bat
python main.py --inspect
python main.py --collect
python main.py --predict
python main.py --backtest
python main.py --dashboard
python main.py --stats
python main.py --run
```

## Tests

```bat
pytest -q
```

## Project layout

See the repository tree under `wingo30_predictor/` for `api/`, `data/`, `models/`, `analysis/`, `scheduler/`, and `tests/`.

## Safety statement

The system is a statistical analyzer. It measures real historical performance and must never claim that an algorithm can know the next random result with 100% certainty.
