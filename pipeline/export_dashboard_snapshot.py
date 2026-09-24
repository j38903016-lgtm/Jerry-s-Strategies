#!/usr/bin/env python3
"""Export the latest complete dashboard run into a compact SQLite snapshot."""
from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path


EXPECTED_SYMBOL_COUNT = 203
DEFAULT_SOURCE = Path(os.environ.get("PIPELINE_DB", "/data/us_stock_pipeline/us_stock_pipeline.sqlite3"))
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "dashboard.sqlite3"


SCHEMA = """
CREATE TABLE pipeline_runs(
  run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
  status TEXT NOT NULL, mode TEXT NOT NULL, gpu_info TEXT, error TEXT
);
CREATE TABLE model_runs(
  run_id TEXT NOT NULL, symbol TEXT NOT NULL, model_type TEXT NOT NULL,
  train_start TEXT, train_end TEXT, test_start TEXT, test_end TEXT,
  n_train INTEGER, n_test INTEGER, best_params TEXT, best_n_estimators INTEGER,
  test_total_return REAL, test_annual_return REAL, test_max_drawdown REAL,
  test_sharpe REAL, strict_buy_precision REAL, strict_sell_precision REAL,
  tolerance_buy_precision REAL, tolerance_sell_precision REAL,
  signal_count INTEGER, trade_count INTEGER, created_at TEXT NOT NULL,
  PRIMARY KEY(run_id,symbol)
);
CREATE TABLE predictions(
  run_id TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL,
  open REAL, close REAL, true_label INTEGER, predicted_signal INTEGER,
  buy_probability REAL, sell_probability REAL, dataset TEXT NOT NULL,
  PRIMARY KEY(run_id,symbol,date,dataset)
);
CREATE INDEX idx_predictions_symbol_date ON predictions(symbol,date);
CREATE TABLE raw_daily(
  symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL,
  close REAL, volume REAL, change_pct REAL, fetched_at TEXT NOT NULL,
  PRIMARY KEY(symbol,date)
);
CREATE INDEX idx_raw_symbol_date ON raw_daily(symbol,date);
"""


def latest_publishable_run(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        """
        SELECT pr.run_id
        FROM pipeline_runs pr
        JOIN (
          SELECT run_id, COUNT(DISTINCT symbol) AS model_count
          FROM model_runs GROUP BY run_id
        ) mr ON mr.run_id=pr.run_id
        JOIN (
          SELECT run_id, COUNT(DISTINCT symbol) AS live_count
          FROM predictions WHERE dataset='live' GROUP BY run_id
        ) lp ON lp.run_id=pr.run_id
        WHERE pr.status='success' AND pr.mode='full'
          AND mr.model_count>=? AND lp.live_count>=?
        ORDER BY julianday(pr.finished_at) DESC
        LIMIT 1
        """,
        (EXPECTED_SYMBOL_COUNT, EXPECTED_SYMBOL_COUNT),
    ).fetchone()
    if row is None:
        raise RuntimeError("No complete publishable dashboard run was found")
    return str(row[0])


def copy_query(
    source: sqlite3.Connection,
    destination: sqlite3.Connection,
    table: str,
    sql: str,
    params: tuple = (),
) -> int:
    rows = source.execute(sql, params).fetchall()
    if not rows:
        return 0
    placeholders = ",".join("?" for _ in rows[0])
    destination.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
    return len(rows)


def export_snapshot(source_path: Path, output_path: Path, run_id: str | None = None) -> dict[str, int | str]:
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)

    source_uri = f"file:{source_path.as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source:
        selected_run_id = run_id or latest_publishable_run(source)
        with sqlite3.connect(temporary_path) as destination:
            destination.executescript(SCHEMA)
            counts = {
                "pipeline_runs": copy_query(
                    source,
                    destination,
                    "pipeline_runs",
                    """
                    SELECT * FROM pipeline_runs
                    WHERE run_id IN (
                      SELECT run_id FROM pipeline_runs
                      ORDER BY julianday(started_at) DESC LIMIT 50
                    ) OR run_id=?
                    """,
                    (selected_run_id,),
                ),
                "model_runs": copy_query(
                    source,
                    destination,
                    "model_runs",
                    "SELECT * FROM model_runs WHERE run_id=? ORDER BY symbol",
                    (selected_run_id,),
                ),
                "predictions": copy_query(
                    source,
                    destination,
                    "predictions",
                    "SELECT * FROM predictions WHERE run_id=? AND dataset IN ('test','live') ORDER BY symbol,date,dataset",
                    (selected_run_id,),
                ),
                "raw_daily": copy_query(
                    source,
                    destination,
                    "raw_daily",
                    """
                    SELECT r.* FROM raw_daily r
                    JOIN (
                      SELECT symbol, MAX(date) AS max_date
                      FROM raw_daily GROUP BY symbol
                    ) latest ON latest.symbol=r.symbol AND latest.max_date=r.date
                    ORDER BY r.symbol
                    """,
                ),
            }
            destination.commit()
            integrity = destination.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(f"Snapshot integrity check failed: {integrity}")
            destination.execute("VACUUM")

    temporary_path.replace(output_path)
    return {"run_id": selected_run_id, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(export_snapshot(args.source, args.output, args.run_id))


if __name__ == "__main__":
    main()
