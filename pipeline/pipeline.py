#!/usr/bin/env python3
"""US-stock daily pipeline: Alpha Vantage -> features -> inflections -> complex XGB.

The modeling flow follows the complex notebook block in 0630.ipynb.  The
database is the source of truth; CSV files are human-readable exports.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import optuna
    import xgboost as xgb
    from alpha_vantage.timeseries import TimeSeries
except ImportError as exc:
    raise SystemExit(f"missing dependency: {exc}") from exc

from sklearn.model_selection import TimeSeriesSplit


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("us_stock_pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)

PIPELINE_HOME = Path(os.environ.get("PIPELINE_HOME", "/data/us_stock_pipeline"))
DB_PATH = Path(os.environ.get("PIPELINE_DB", PIPELINE_HOME / "us_stock_pipeline.sqlite3"))
EXPORT_ROOT = Path(os.environ.get("PIPELINE_EXPORTS", PIPELINE_HOME / "exports"))
DATA_YEARS = int(os.environ.get("DATA_YEARS", "5"))
N_WINDOW = 5
TEST_RATIO = 0.20
CV_SPLITS = 5
OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "15"))
RANDOM_STATE = 42
TOLERANCE = 2
INITIAL_CASH = 10_000_000.0
BUY_FEE_RATE = 0.00031
SELL_FEE_RATE = 0.00081
SLIPPAGE_RATE = 0.001
DELAY_BETWEEN_REQUESTS = float(os.environ.get("AV_DELAY_SECONDS", "13"))
MAX_RETRIES = max(1, int(os.environ.get("AV_REQUEST_RETRIES", "5")))
FAILED_SYMBOL_RETRY_ROUNDS = max(0, int(os.environ.get("FAILED_SYMBOL_RETRY_ROUNDS", "2")))
FAILED_SYMBOL_RETRY_BASE_SECONDS = max(
    0.0, float(os.environ.get("FAILED_SYMBOL_RETRY_BASE_SECONDS", "60"))
)
MODEL_MAX_WORKERS = max(1, int(os.environ.get("MODEL_MAX_WORKERS", "16")))
XGB_N_JOBS = max(1, int(os.environ.get("XGB_N_JOBS", "4")))
MARKET_TZ = ZoneInfo("America/New_York")

FEATURE_COLS = [
    "K_Line_Body", "MACD", "Histogram", "MA5", "MA12", "BIAS12", "RSI",
    "OBV", "Curvature", "Daily_Log_Return", "RW_Deviation",
    "Volume_Spike_Ratio", "PV_Composite_Spike",
]

BASELINE_US_SECTORS = {
    # Original production universe retained unchanged.
    "人工智能": ["MSFT", "GOOGL", "AMZN", "META", "ORCL", "CRM", "PLTR", "NVDA", "AMD", "AVGO"],
    "机器人": ["ISRG", "ROK", "TER", "CGNX", "SYM", "ZBRA", "PATH", "MBLY", "AUR", "ABB"],
    "芯片": ["NVDA", "AMD", "AVGO", "INTC", "QCOM", "MU", "MRVL", "ARM", "TXN", "ADI"],
    "半导体": ["TSM", "ASML", "AMAT", "LRCX", "KLAC", "NXPI", "ON", "MCHP", "MPWR", "TXN"],
    "银行": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF"],
    "黄金": ["NEM", "EGO", "AEM", "KGC", "AU", "GFI", "HMY", "WPM", "FNV", "RGLD"],
    "有色金属": ["FCX", "SCCO", "AA", "CENX", "TECK", "ALB", "SQM", "MP", "LAC", "NEM"],
}

# Added 2026-09-22. The nine new groups complete a compact GICS-style sector
# backbone; each group has 13 representatives so offensive, cyclical and
# defensive exposures have comparable symbol counts. Existing groups receive
# three additions each. Total: 16 groups x 13 slots = 208, 203 unique symbols.
ADDED_US_SECTORS_20260922 = {
    "人工智能": ["ADBE", "NOW", "IBM"],
    "机器人": ["HON", "EMR", "ETN"],
    "芯片": ["SWKS", "QRVO", "MTSI"],
    "半导体": ["GFS", "ENTG", "AMKR"],
    "银行": ["FITB", "STT", "NTRS"],
    "黄金": ["IAG", "AGI", "OR"],
    "有色金属": ["BHP", "RIO", "VALE"],
    "医疗保健": ["LLY", "JNJ", "UNH", "ABBV", "MRK", "PFE", "TMO", "ABT", "AMGN", "GILD", "MDT", "SYK", "BSX"],
    "必需消费": ["WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL", "MDLZ", "KHC", "SYY", "KMB", "KR"],
    "公用事业": ["NEE", "SO", "DUK", "AEP", "SRE", "D", "EXC", "XEL", "ED", "PEG", "WEC", "ES", "AWK"],
    "房地产": ["PLD", "AMT", "EQIX", "WELL", "SPG", "O", "DLR", "PSA", "CCI", "VICI", "ESS", "MAA", "CBRE"],
    "能源": ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "KMI", "WMB", "LNG", "OKE"],
    "可选消费": ["TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG", "ORLY", "AZO", "MAR", "GM"],
    "通信服务": ["NFLX", "DIS", "CMCSA", "T", "TMUS", "VZ", "CHTR", "LYV", "TTWO", "WBD", "SPOT", "FOXA", "PINS"],
    "综合工业/国防": ["GE", "RTX", "LMT", "NOC", "GD", "BA", "CAT", "DE", "UPS", "FDX", "UNP", "CSX", "WM"],
    "保险": ["CB", "BRO", "AON", "PGR", "TRV", "ALL", "MET", "PRU", "AFL", "HIG", "ACGL", "CINF", "AJG"],
}

US_SECTORS = {
    sector: list(BASELINE_US_SECTORS.get(sector, [])) + list(added_symbols)
    for sector, added_symbols in ADDED_US_SECTORS_20260922.items()
}
NEW_STOCK_CODES = sorted({
    symbol for symbols in ADDED_US_SECTORS_20260922.values() for symbol in symbols
})
STOCK_CODES = sorted({symbol for symbols in US_SECTORS.values() for symbol in symbols})

if any(len(symbols) != 13 for symbols in US_SECTORS.values()):
    raise RuntimeError("US universe configuration requires exactly 13 symbols per sector")
if len(US_SECTORS) != 16 or len(STOCK_CODES) != 203:
    raise RuntimeError("US universe configuration must contain 16 sectors and 203 unique symbols")

# ======================== US MARKET ADJUSTMENTS ========================
# These are the only model search-space changes from the A-share source.
# They are centralized so both the batch and daily-test paths stay identical.
# Original -> US:
# percentile_threshold 80-96 -> 82-97; price buy rank 5-25 -> 5-30;
# price sell rank 75-95 -> 70-95; min gap 0.005-0.05 -> 0.0075-0.06;
# estimators 600-1200 -> 650-1300; the remaining lists add nearby candidates
# for the smoother continuous-auction market and larger overnight gaps.
US_THRESHOLD_RANGES = {
    "percentile_threshold": (82.0, 97.0),
    "price_percentile_buy": (5, 30),
    "price_percentile_sell": (70, 95),
    "min_price_gap": (0.0075, 0.06),
}
US_MODEL_SEARCH_SPACE = {
    "n_estimators": (650, 1300),
    "learning_rate": [0.01, 0.015, 0.02, 0.03],
    "max_depth": [2, 3, 4],
    "gamma": [0.5, 1.0, 1.5, 2.0],
    "min_child_weight": [8, 12, 20],
    "max_delta_step": [1, 2],
    "subsample": [0.65, 0.8, 0.9],
    "colsample_bytree": [0.55, 0.7, 0.85],
    "reg_lambda": [5.0, 8.0, 12.0],
    "reg_alpha": [0.5, 1.5, 3.0],
}
# ====================== END US MARKET ADJUSTMENTS ======================



def ensure_dirs() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)


def connect_db():
    import sqlite3
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS pipeline_runs(
      run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
      status TEXT NOT NULL, mode TEXT NOT NULL, gpu_info TEXT, error TEXT);
    CREATE TABLE IF NOT EXISTS raw_daily(
      symbol TEXT NOT NULL, date TEXT NOT NULL, open REAL, high REAL, low REAL,
      close REAL, volume REAL, change_pct REAL, fetched_at TEXT NOT NULL,
      PRIMARY KEY(symbol,date));
    CREATE TABLE IF NOT EXISTS features(
      symbol TEXT NOT NULL, date TEXT NOT NULL, "K_Line_Body" REAL,
      "MACD" REAL, "Histogram" REAL, "MA5" REAL, "MA12" REAL, "BIAS12" REAL,
      "RSI" REAL, "OBV" REAL, "Curvature" REAL, "Daily_Log_Return" REAL,
      "RW_Deviation" REAL, "Volume_Spike_Ratio" REAL,
      "PV_Composite_Spike" REAL, run_id TEXT NOT NULL, PRIMARY KEY(symbol,date));
    CREATE TABLE IF NOT EXISTS inflection_labels(
      symbol TEXT NOT NULL, date TEXT NOT NULL, label INTEGER, run_id TEXT NOT NULL,
      PRIMARY KEY(symbol,date));
    CREATE TABLE IF NOT EXISTS cv_folds(
      run_id TEXT NOT NULL, symbol TEXT NOT NULL, fold INTEGER NOT NULL,
      score REAL, best_n_estimators INTEGER, PRIMARY KEY(run_id,symbol,fold));
    CREATE TABLE IF NOT EXISTS model_runs(
      run_id TEXT NOT NULL, symbol TEXT NOT NULL, model_type TEXT NOT NULL,
      train_start TEXT, train_end TEXT, test_start TEXT, test_end TEXT,
      n_train INTEGER, n_test INTEGER, best_params TEXT, best_n_estimators INTEGER,
      test_total_return REAL, test_annual_return REAL, test_max_drawdown REAL,
      test_sharpe REAL, strict_buy_precision REAL, strict_sell_precision REAL,
      tolerance_buy_precision REAL, tolerance_sell_precision REAL,
      signal_count INTEGER, trade_count INTEGER, created_at TEXT NOT NULL,
      PRIMARY KEY(run_id,symbol));
    CREATE TABLE IF NOT EXISTS predictions(
      run_id TEXT NOT NULL, symbol TEXT NOT NULL, date TEXT NOT NULL,
      open REAL, close REAL, true_label INTEGER, predicted_signal INTEGER,
      buy_probability REAL, sell_probability REAL, dataset TEXT NOT NULL,
      PRIMARY KEY(run_id,symbol,date,dataset));
    CREATE INDEX IF NOT EXISTS idx_raw_symbol_date ON raw_daily(symbol,date);
    CREATE INDEX IF NOT EXISTS idx_predictions_symbol_date ON predictions(symbol,date);
    """)
    conn.commit()


def market_now_iso() -> str:
    return datetime.now(MARKET_TZ).isoformat()


def completed_data_cutoff() -> pd.Timestamp:
    """Return the exclusive date cutoff for completed New York trading days."""
    return pd.Timestamp(datetime.now(MARKET_TZ).date())


def assert_completed_data(raw: pd.DataFrame, symbol: str) -> None:
    cutoff = completed_data_cutoff()
    if not raw.empty and raw["Date"].max() >= cutoff:
        raise ValueError(
            f"{symbol}: data contains current New York date or later "
            f"({raw['Date'].max().date()} >= {cutoff.date()})"
        )


def gpu_info() -> str:
    failures = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("REQUIRE_GPU", "1") == "1":
            raise RuntimeError("nvidia-smi unavailable; GPU is required") from exc
        return "GPU unavailable; CPU fallback enabled"


def xgb_gpu_params() -> dict:
    if os.environ.get("REQUIRE_GPU", "1") != "1":
        return {"tree_method": "hist", "n_jobs": XGB_N_JOBS}
    version = tuple(int(part) for part in xgb.__version__.split(".")[:2])
    if version >= (2, 0):
        return {"tree_method": "hist", "device": "cuda", "n_jobs": XGB_N_JOBS}
    return {"tree_method": "hist", "n_jobs": XGB_N_JOBS}


def make_model(params: dict):
    return xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        random_state=RANDOM_STATE, early_stopping_rounds=15,
        eval_metric="mlogloss", **xgb_gpu_params(), **params,
    )


def strip_suffix(symbol: str) -> str:
    return symbol.split(".")[0]


def redact_secret(value: object, secret: str | None) -> str:
    text = str(value)
    return text.replace(secret, "***") if secret else text


def fetch_data(symbol: str, api_key: str, outputsize: str = "full") -> pd.DataFrame:
    # US DATA ADJUSTMENT: Alpha Vantage accepts US-listed tickers directly.
    av_symbol = symbol
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            ts = TimeSeries(key=api_key, output_format="pandas")
            frame, _ = ts.get_daily_adjusted(symbol=av_symbol, outputsize=outputsize)
            if frame is None or frame.empty:
                raise ValueError("empty Alpha Vantage response")
            frame = frame.reset_index()
            rename = {"date": "Date", "1. open": "Open", "2. high": "High",
                      "3. low": "Low", "4. close": "Close", "6. volume": "Volume"}
            frame = frame.rename(columns=rename)
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.loc[frame["Date"] < completed_data_cutoff()]
            if outputsize == "full":
                cutoff = completed_data_cutoff() - pd.Timedelta(days=DATA_YEARS * 365)
                frame = frame.loc[frame["Date"] >= cutoff]
            frame = frame[["Date", "Open", "High", "Low", "Close", "Volume"]]
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame = frame.dropna(subset=["Date", "Open", "Close", "Volume"])
            frame = (
                frame.sort_values("Date", ascending=True)
                .drop_duplicates("Date")
                .reset_index(drop=True)
            )
            frame["ChangePct"] = frame["Close"].pct_change().mul(100).fillna(0)
            return frame
        except Exception as exc:
            last_error = exc
            msg = str(exc).lower()
            if attempt >= MAX_RETRIES - 1:
                break
            if any(token in msg for token in ("frequency", "limit", "rate", "thank you")):
                wait = 60 + attempt * 30
            else:
                wait = min(15 * (2 ** attempt), 120)
            log.warning(
                "%s fetch attempt %d/%d failed: %s; retrying in %ss",
                symbol,
                attempt + 1,
                MAX_RETRIES,
                redact_secret(exc, api_key),
                wait,
            )
            time.sleep(wait)
    safe_error = redact_secret(last_error, api_key)
    raise RuntimeError(f"fetch failed for {symbol}: {safe_error}") from last_error


def clean_volume(value) -> float:
    text = str(value).strip().upper().replace(",", "")
    try:
        if text.endswith("M"): return float(text[:-1]) * 1e6
        if text.endswith("K"): return float(text[:-1]) * 1e3
        return float(text)
    except ValueError:
        return 0.0


def compute_features(raw: pd.DataFrame) -> pd.DataFrame:
    c = raw["Close"].astype(float)
    o = raw["Open"].astype(float)
    v = raw["Volume"].astype(float).map(clean_volume)
    out = pd.DataFrame(index=raw.index)
    out["K_Line_Body"] = c / o - 1
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    signal = out["MACD"].ewm(span=9, adjust=False).mean()
    out["Histogram"] = 2 * (out["MACD"] - signal)
    out["MA5"], out["MA12"] = c.rolling(5).mean(), c.rolling(12).mean()
    out["BIAS12"] = (c - out["MA12"]) / out["MA12"] * 100
    delta = c.diff(); up = delta.clip(lower=0); down = (-delta).clip(lower=0)
    rs = up.rolling(14).sum() / (down.rolling(14).sum() + 1e-9)
    out["RSI"] = 100 - 100 / (1 + rs)
    out["OBV"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    dy = c.diff().fillna(0); ddy = dy.diff().fillna(0)
    out["Curvature"] = ddy / (1 + dy ** 2) ** 1.5
    log_return = np.log(c / c.shift(1)); out["Daily_Log_Return"] = log_return
    t = 7; out["RW_Deviation"] = np.log(c / c.shift(t)) / (log_return.rolling(t).std() * np.sqrt(t) + 1e-9)
    w = 3; mean = log_return.rolling(w).mean(); std = log_return.rolling(w).std()
    z = (log_return - mean) / (std + 1e-9)
    out["Volume_Spike_Ratio"] = (v / (v.rolling(w).median() + 1e-9)).clip(upper=10)
    out["PV_Composite_Spike"] = (np.sign(log_return) * z.abs() * out["Volume_Spike_Ratio"]).clip(-20, 20)
    return out.fillna(0)


def compute_labels(raw: pd.DataFrame) -> pd.Series:
    labels = np.zeros(len(raw), dtype=float)
    for i in range(len(raw)):
        if i < N_WINDOW or i + N_WINDOW >= len(raw):
            labels[i] = np.nan
            continue
        window = raw["Close"].iloc[i - N_WINDOW:i + N_WINDOW + 1]
        if raw["Close"].iloc[i] == window.min(): labels[i] = -1
        elif raw["Close"].iloc[i] == window.max(): labels[i] = 1
    return pd.Series(labels, index=raw.index, name="label")


def safe_best_n_estimators(model) -> int:
    best = getattr(model, "best_iteration", None)
    return int(best + 1) if best is not None else int(model.get_params().get("n_estimators", 100))


def make_purged_time_series_splits(X, n_splits=5, purge_window=N_WINDOW):
    splitter = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, val_idx in splitter.split(X):
        val_start = val_idx[0]
        clean_train_idx = train_idx[train_idx < val_start - purge_window]
        if len(clean_train_idx) == 0:
            continue
        yield clean_train_idx, val_idx


def split_calibration_and_evaluation(val_idx, calibration_ratio=0.50, min_eval_size=20):
    cut = int(len(val_idx) * calibration_ratio)
    if cut <= N_WINDOW or len(val_idx) - cut < min_eval_size:
        return None, None, None

    calibration_idx = val_idx[:cut]
    evaluation_idx = val_idx[cut:]
    eval_start = evaluation_idx[0]

    calibration_fit_idx = calibration_idx[calibration_idx < eval_start - N_WINDOW]
    if len(calibration_fit_idx) < min_eval_size:
        return None, None, None
    return calibration_fit_idx, calibration_idx, evaluation_idx


def generate_signals_simple(
    val_proba,
    close_prices,
    thresh_buy,
    thresh_sell,
    N=N_WINDOW,
    price_percentile_buy=15,
    price_percentile_sell=85,
    min_price_gap=0.02,
):
    preds = np.zeros(len(val_proba), dtype=int)
    state = 0
    buy_price = 0.0

    for i in range(len(val_proba)):
        if i < N:
            continue

        start = max(0, i - N)
        end = i + 1
        prob_buy = val_proba[i, 1]
        prob_sell = val_proba[i, 2]

        is_buy_peak = prob_buy >= np.max(val_proba[start:end, 1])
        is_sell_peak = prob_sell >= np.max(val_proba[start:end, 2])

        win_prices = close_prices[start:end]
        if len(win_prices) <= 1:
            continue
        price_rank = (np.sum(win_prices <= close_prices[i]) - 1) / (len(win_prices) - 1) * 100

        final_buy = (
            prob_buy >= thresh_buy
            and price_rank <= price_percentile_buy
            and is_buy_peak
        )
        final_sell = (
            prob_sell >= thresh_sell
            and price_rank >= price_percentile_sell
            and is_sell_peak
        )

        if state == 0 and final_buy:
            preds[i] = 1
            state = 1
            buy_price = close_prices[i]
        elif state == 1 and final_sell and (close_prices[i] / buy_price - 1) >= min_price_gap:
            preds[i] = 2
            state = 0
            buy_price = 0.0

    return preds


def run_next_open_backtest(
    signals,
    prices_df,
    initial_cash=INITIAL_CASH,
    buy_fee_rate=BUY_FEE_RATE,
    sell_fee_rate=SELL_FEE_RATE,
    slippage_rate=SLIPPAGE_RATE,
):
    prices_df = prices_df.reset_index(drop=True)
    signals = np.asarray(signals, dtype=int)

    cash = float(initial_cash)
    shares = 0.0
    position = 0
    pending_signal = 0
    net_values = []
    trades = []

    for t in range(len(prices_df)):
        today = prices_df.loc[t]
        open_price = float(today['Open'])
        close_price = float(today['Close'])

        if pending_signal == 2 and position == 1:
            sell_price = open_price * (1 - slippage_rate)
            cash += shares * sell_price * (1 - sell_fee_rate)
            trades.append({'Date': today['Date'], 'Action': 'SELL', 'Price': sell_price, 'Shares': shares})
            shares = 0.0
            position = 0
        elif pending_signal == 1 and position == 0:
            buy_price = open_price * (1 + slippage_rate)
            per_cost = buy_price * (1 + buy_fee_rate)
            allow = int((cash / per_cost) // 100) * 100
            if allow > 0:
                shares = float(allow)
                cash -= shares * per_cost
                position = 1
                trades.append({'Date': today['Date'], 'Action': 'BUY', 'Price': buy_price, 'Shares': shares})

        net_values.append((cash + shares * close_price) / initial_cash)
        pending_signal = signals[t]

    return np.asarray(net_values, dtype=float), pd.DataFrame(trades)


def summarize_net_values(net_values):
    net_values = pd.Series(net_values, dtype=float)
    if len(net_values) == 0:
        return {'total_return': 0.0, 'annual_return': 0.0, 'max_dd': 0.0, 'sharpe': 0.0}

    total_return = float(net_values.iloc[-1] - 1.0)
    annual_return = float(net_values.iloc[-1] ** (252.0 / len(net_values)) - 1.0) if net_values.iloc[-1] > 0 else -1.0
    drawdown = (net_values - net_values.cummax()) / net_values.cummax()
    max_dd = float(drawdown.min())
    rets = net_values.pct_change().dropna()
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    return {'total_return': total_return, 'annual_return': annual_return, 'max_dd': max_dd, 'sharpe': sharpe}


def exact_precision(y_true, y_pred, target_label):
    pred_idx = np.where(y_pred == target_label)[0]
    if len(pred_idx) == 0:
        return 0.0
    return float(np.mean(y_true[pred_idx] == target_label))


def tolerance_precision_one_to_one(y_true, y_pred, target_label, tol=TOLERANCE):
    true_idx = list(np.where(y_true == target_label)[0])
    pred_idx = np.where(y_pred == target_label)[0]
    if len(pred_idx) == 0:
        return 0.0

    used_true = set()
    hits = 0
    for p in pred_idx:
        candidates = [t for t in true_idx if t not in used_true and abs(t - p) <= tol]
        if candidates:
            chosen = min(candidates, key=lambda t: abs(t - p))
            used_true.add(chosen)
            hits += 1
    return hits / len(pred_idx)


def db_raw(conn, symbol: str, raw: pd.DataFrame, fetched_at: str) -> None:
    conn.execute("DELETE FROM raw_daily WHERE symbol = ?", (symbol,))
    rows = [(symbol, row.Date.strftime("%Y-%m-%d"), float(row.Open), float(row.High), float(row.Low),
             float(row.Close), float(row.Volume), float(row.ChangePct), fetched_at)
             for row in raw.itertuples()]
    conn.executemany("INSERT OR REPLACE INTO raw_daily VALUES (?,?,?,?,?,?,?,?,?)", rows)


def load_raw_from_db(conn, symbol: str) -> pd.DataFrame:
    raw = pd.read_sql_query(
        "SELECT date AS Date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume, change_pct AS ChangePct "
        "FROM raw_daily WHERE symbol = ? ORDER BY date",
        conn,
        params=(symbol,),
    )
    if raw.empty:
        raise ValueError(f"{symbol}: no cached raw data")
    raw["Date"] = pd.to_datetime(raw["Date"])
    return raw


def refresh_raw_incremental(conn, symbol: str, api_key: str) -> tuple[pd.DataFrame, bool]:
    """Fetch only the compact API window and keep each symbol's row count fixed."""
    try:
        cached = load_raw_from_db(conn, symbol)
    except ValueError:
        return fetch_data(symbol, api_key, outputsize="full"), True

    previous_count = len(cached)
    cached = cached.loc[cached["Date"] < completed_data_cutoff()].copy()
    latest_window = fetch_data(symbol, api_key, outputsize="compact")
    if latest_window.empty:
        return cached.tail(previous_count).reset_index(drop=True), len(cached) != previous_count
    old_latest = cached["Date"].max() if not cached.empty else pd.Timestamp.min
    combined = pd.concat([cached, latest_window], ignore_index=True)
    combined = combined.sort_values("Date", ascending=True).drop_duplicates("Date", keep="last").reset_index(drop=True)
    changed = combined["Date"].max() > old_latest or len(cached) != previous_count
    combined["ChangePct"] = combined["Close"].pct_change().mul(100).fillna(0)
    combined = combined.tail(previous_count).reset_index(drop=True)
    combined = combined.sort_values('Date', ascending=True)
    return combined, bool(changed)


def db_features_labels(conn, run_id: str, symbol: str, raw: pd.DataFrame, feats: pd.DataFrame, labels: pd.Series) -> None:
    conn.execute("DELETE FROM features WHERE symbol = ?", (symbol,))
    conn.execute("DELETE FROM inflection_labels WHERE symbol = ?", (symbol,))
    feature_rows = []
    label_rows = []
    for i, row in raw.iterrows():
        date = row.Date.strftime("%Y-%m-%d")
        feature_rows.append((symbol, date, *[float(feats.iloc[i][c]) for c in FEATURE_COLS], run_id))
        label = labels.iloc[i]
        label_rows.append((symbol, date, None if pd.isna(label) else int(label), run_id))
    conn.executemany("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", feature_rows)
    conn.executemany("INSERT OR REPLACE INTO inflection_labels VALUES (?,?,?,?)", label_rows)


def run_model(
    conn,
    run_id: str,
    symbol: str,
    raw: pd.DataFrame,
    feats: pd.DataFrame,
    labels: pd.Series,
) -> None:
    source_df = pd.concat([raw[["Date", "Open", "Close"]], feats, labels], axis=1)
    valid_rows = source_df["label"].notna()
    model_df = source_df.loc[valid_rows].reset_index(drop=True)

    X_all = model_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y_all = model_df["label"].map({0: 0, -1: 1, 1: 2}).astype(int)
    prices_all = model_df[["Date", "Open", "Close"]].copy()

    raw_split_idx = int(len(X_all) * (1 - TEST_RATIO))
    train_val_end = raw_split_idx - N_WINDOW
    if train_val_end <= 0:
        raise ValueError(f"{symbol}样本过少，无法隔离训练测试")

    X_train_val = X_all.iloc[:train_val_end].reset_index(drop=True)
    y_train_val = y_all.iloc[:train_val_end].reset_index(drop=True)
    prices_train_val = prices_all.iloc[:train_val_end].reset_index(drop=True)
    X_test = X_all.iloc[raw_split_idx:].reset_index(drop=True)
    y_test = y_all.iloc[raw_split_idx:].reset_index(drop=True).values
    prices_test = prices_all.iloc[raw_split_idx:].reset_index(drop=True)

    def objective(trial):
        percentile_threshold = trial.suggest_float('percentile_threshold', *US_THRESHOLD_RANGES['percentile_threshold'])
        price_percentile_buy = trial.suggest_int('price_percentile_buy', *US_THRESHOLD_RANGES['price_percentile_buy'])
        price_percentile_sell = trial.suggest_int('price_percentile_sell', *US_THRESHOLD_RANGES['price_percentile_sell'])
        min_price_gap = trial.suggest_float('min_price_gap', *US_THRESHOLD_RANGES['min_price_gap'])

        params = {
            'n_estimators': trial.suggest_int('n_estimators', *US_MODEL_SEARCH_SPACE['n_estimators']),
            'learning_rate': trial.suggest_categorical('learning_rate', US_MODEL_SEARCH_SPACE['learning_rate']),
            'max_depth': trial.suggest_categorical('max_depth', US_MODEL_SEARCH_SPACE['max_depth']),
            'gamma': trial.suggest_categorical('gamma', US_MODEL_SEARCH_SPACE['gamma']),
            'min_child_weight': trial.suggest_categorical('min_child_weight', US_MODEL_SEARCH_SPACE['min_child_weight']),
            'max_delta_step': trial.suggest_categorical('max_delta_step', US_MODEL_SEARCH_SPACE['max_delta_step']),
            'subsample': trial.suggest_categorical('subsample', US_MODEL_SEARCH_SPACE['subsample']),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', US_MODEL_SEARCH_SPACE['colsample_bytree']),
            'reg_lambda': trial.suggest_categorical('reg_lambda', US_MODEL_SEARCH_SPACE['reg_lambda']),
            'reg_alpha': trial.suggest_categorical('reg_alpha', US_MODEL_SEARCH_SPACE['reg_alpha'])
        }

        fold_scores = []
        fold_records = []

        for fold, (train_idx, val_idx) in enumerate(make_purged_time_series_splits(X_train_val, CV_SPLITS, N_WINDOW), 1):
            calibration_fit_idx, calibration_idx, evaluation_idx = split_calibration_and_evaluation(val_idx)
            if calibration_fit_idx is None:
                continue

            X_train = X_train_val.iloc[train_idx]
            y_train = y_train_val.iloc[train_idx]
            X_cal_fit = X_train_val.iloc[calibration_fit_idx]
            y_cal_fit = y_train_val.iloc[calibration_fit_idx]
            model = make_model(params)
            model.fit(X_train, y_train, eval_set=[(X_cal_fit, y_cal_fit)], verbose=False)
            calibration_proba = model.predict_proba(X_train_val.iloc[calibration_idx])
            thresh_buy = np.percentile(calibration_proba[:, 1], percentile_threshold)
            thresh_sell = np.percentile(calibration_proba[:, 2], percentile_threshold)
            eval_proba = model.predict_proba(X_train_val.iloc[evaluation_idx])
            eval_prices = prices_train_val.iloc[evaluation_idx].reset_index(drop=True)
            eval_preds = generate_signals_simple(
                eval_proba,
                eval_prices['Close'].values,
                thresh_buy,
                thresh_sell,
                N=N_WINDOW,
                price_percentile_buy=price_percentile_buy,
                price_percentile_sell=price_percentile_sell,
                min_price_gap=min_price_gap
            )

            net_values, trades = run_next_open_backtest(eval_preds, eval_prices)
            metrics = summarize_net_values(net_values)
            if trades.empty:
                fold_scores.append(-1.0)
                continue
            score = metrics['sharpe'] + 0.25 * metrics['annual_return'] + metrics['total_return'] + metrics['max_dd']
            fold_scores.append(score)
            fold_records.append((fold, float(score), safe_best_n_estimators(model)))

        trial.set_user_attr("fold_records", fold_records)
        return float(np.mean(fold_scores))


    print('=====================Optuna starting...========================')
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print('========================Optuna end.======================')

    conn.execute("DELETE FROM cv_folds WHERE run_id=? AND symbol=?", (run_id, symbol))
    conn.executemany(
        "INSERT OR REPLACE INTO cv_folds VALUES (?,?,?,?,?)",
        [
            (run_id, symbol, fold, score, best_trees)
            # The sequential pipeline left cv_folds containing the last trial's
            # records because every trial replaced the same symbol/fold keys.
            # Preserve that database contract while writing only once.
            for fold, score, best_trees in study.trials[-1].user_attrs.get("fold_records", [])
        ],
    )

    best_params = study.best_params.copy()
    best_percentile = best_params.pop('percentile_threshold')
    best_price_buy = best_params.pop('price_percentile_buy')
    best_price_sell = best_params.pop('price_percentile_sell')
    best_gap = best_params.pop('min_price_gap')

    early_start = int(len(X_train_val) * 0.60)
    threshold_start = int(len(X_train_val) * 0.82)
    train_end = early_start - N_WINDOW
    early_end = threshold_start - N_WINDOW
    X_model_train = X_train_val.iloc[:train_end]
    y_model_train = y_train_val.iloc[:train_end]
    X_early_stop = X_train_val.iloc[early_start:early_end]
    y_early_stop = y_train_val.iloc[early_start:early_end]
    X_threshold_cal = X_train_val.iloc[threshold_start:]

    final_model = make_model(best_params)
    final_model.fit(
        X_model_train,
        y_model_train,
        eval_set=[(X_model_train, y_model_train), (X_early_stop, y_early_stop)],
        verbose=False
    )
    final_best_trees = safe_best_n_estimators(final_model)

    cal_proba = final_model.predict_proba(X_threshold_cal)
    final_thresh_buy = np.percentile(cal_proba[:, 1], best_percentile)
    final_thresh_sell = np.percentile(cal_proba[:, 2], best_percentile)
    test_proba = final_model.predict_proba(X_test)
    test_preds = generate_signals_simple(
        test_proba,
        prices_test['Close'].values,
        final_thresh_buy,
        final_thresh_sell,
        N=N_WINDOW,
        price_percentile_buy=best_price_buy,
        price_percentile_sell=best_price_sell,
        min_price_gap=best_gap
    )
    labeled_positions = np.flatnonzero(source_df["label"].notna().to_numpy())
    live_context = source_df.iloc[labeled_positions[-1] + 1:].copy()
    live_prediction_row = None
    if not live_context.empty:
        X_live = live_context[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        live_proba = final_model.predict_proba(X_live)
        combined_proba = np.vstack([test_proba, live_proba])
        combined_close = np.concatenate([
            prices_test["Close"].to_numpy(dtype=float),
            live_context["Close"].to_numpy(dtype=float),
        ])
        combined_preds = generate_signals_simple(
            combined_proba,
            combined_close,
            final_thresh_buy,
            final_thresh_sell,
            N=N_WINDOW,
            price_percentile_buy=best_price_buy,
            price_percentile_sell=best_price_sell,
            min_price_gap=best_gap,
        )
        if not np.array_equal(combined_preds[:len(test_preds)], test_preds):
            raise RuntimeError(f"{symbol}: live context changed historical test signals")
        latest_live = live_context.iloc[-1]
        live_prediction_row = (
            run_id,
            symbol,
            pd.Timestamp(latest_live["Date"]).strftime('%Y-%m-%d'),
            float(latest_live["Open"]),
            float(latest_live["Close"]),
            None,
            int(combined_preds[-1]),
            float(live_proba[-1, 1]),
            float(live_proba[-1, 2]),
            'live',
        )
    net_values, trades = run_next_open_backtest(test_preds, prices_test)
    metrics = summarize_net_values(net_values)
    strict_buy = exact_precision(y_test, test_preds, target_label=1)
    strict_sell = exact_precision(y_test, test_preds, target_label=2)
    tol_buy = tolerance_precision_one_to_one(y_test, test_preds, target_label=1, tol=TOLERANCE)
    tol_sell = tolerance_precision_one_to_one(y_test, test_preds, target_label=2, tol=TOLERANCE)

    model_params = {
        **best_params,
        'percentile_threshold': best_percentile,
        'price_percentile_buy': best_price_buy,
        'price_percentile_sell': best_price_sell,
        'min_price_gap': best_gap,
        'final_buy_probability_threshold': float(final_thresh_buy),
        'final_sell_probability_threshold': float(final_thresh_sell),
    }
    row = (
        run_id, symbol, 'complex_xgb_gpu_us_adjusted',
        model_df.Date.iloc[0].strftime('%Y-%m-%d'),
        model_df.Date.iloc[train_val_end - 1].strftime('%Y-%m-%d'),
        prices_test.Date.iloc[0].strftime('%Y-%m-%d'),
        prices_test.Date.iloc[-1].strftime('%Y-%m-%d'),
        len(X_train_val), len(X_test), json.dumps(model_params, ensure_ascii=False), final_best_trees,
        metrics['total_return'], metrics['annual_return'], metrics['max_dd'], metrics['sharpe'],
        exact_precision(y_test, test_preds, 1), exact_precision(y_test, test_preds, 2),
        tolerance_precision_one_to_one(y_test, test_preds, 1, tol=TOLERANCE),
        tolerance_precision_one_to_one(y_test, test_preds, 2, tol=TOLERANCE),
        int(np.sum(test_preds != 0)), len(trades), market_now_iso()
    )
    conn.execute("INSERT OR REPLACE INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    pred_rows = [
        (
            run_id, symbol, prices_test.Date.iloc[i].strftime('%Y-%m-%d'),
            float(prices_test.Open.iloc[i]), float(prices_test.Close.iloc[i]), int(y_test[i]),
            int(test_preds[i]), float(test_proba[i, 1]), float(test_proba[i, 2]), 'test'
        )
        for i in range(len(test_preds))
    ]
    conn.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)", pred_rows)
    if live_prediction_row is not None:
        conn.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)", live_prediction_row)
    out_dir = EXPORT_ROOT / symbol.replace('.', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'Date': prices_test.Date,
        'Open': prices_test.Open,
        'Close': prices_test.Close,
        '真实拐点编码': y_test,
        '预测信号': test_preds,
        '买入概率': test_proba[:, 1],
        '卖出概率': test_proba[:, 2],
    }).to_csv(out_dir / f'{strip_suffix(symbol)}_测试集预测标签.csv', index=False, encoding='utf-8-sig')
    log.info(
        '%s model done: test=%d signals=%d trades=%d return=%.2f%% live_date=%s live_signal=%s',
        symbol,
        len(test_preds),
        int(np.sum(test_preds != 0)),
        len(trades),
        metrics['total_return'] * 100,
        live_prediction_row[2] if live_prediction_row is not None else None,
        live_prediction_row[6] if live_prediction_row is not None else None,
    )


def run_model_daily_test(
    conn,
    run_id: str,
    symbol: str,
    raw: pd.DataFrame,
    feats: pd.DataFrame,
    labels: pd.Series,
) -> None:
    source_df = pd.concat([raw[["Date", "Open", "Close"]], feats, labels], axis=1)
    valid_rows = source_df["label"].notna()
    model_df = source_df.loc[valid_rows].reset_index(drop=True)

    X_all = model_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
    y_all = model_df["label"].map({0: 0, -1: 1, 1: 2}).astype(int)
    prices_all = model_df[["Date", "Open", "Close"]].copy()

    raw_split_idx = int(len(X_all) * (1 - TEST_RATIO))
    train_val_end = raw_split_idx - N_WINDOW
    if train_val_end <= 0:
        raise ValueError(f"{symbol}样本过少，无法隔离训练测试")

    X_train_val = X_all.iloc[:train_val_end].reset_index(drop=True)
    y_train_val = y_all.iloc[:train_val_end].reset_index(drop=True)
    prices_train_val = prices_all.iloc[:train_val_end].reset_index(drop=True)
    X_test = X_all.iloc[raw_split_idx:].reset_index(drop=True)
    y_test = y_all.iloc[raw_split_idx:].reset_index(drop=True).values
    prices_test = prices_all.iloc[raw_split_idx:].reset_index(drop=True)

    def objective(trial):
        percentile_threshold = trial.suggest_float('percentile_threshold', *US_THRESHOLD_RANGES['percentile_threshold'])
        price_percentile_buy = trial.suggest_int('price_percentile_buy', *US_THRESHOLD_RANGES['price_percentile_buy'])
        price_percentile_sell = trial.suggest_int('price_percentile_sell', *US_THRESHOLD_RANGES['price_percentile_sell'])
        min_price_gap = trial.suggest_float('min_price_gap', *US_THRESHOLD_RANGES['min_price_gap'])

        params = {
            'n_estimators': trial.suggest_int('n_estimators', *US_MODEL_SEARCH_SPACE['n_estimators']),
            'learning_rate': trial.suggest_categorical('learning_rate', US_MODEL_SEARCH_SPACE['learning_rate']),
            'max_depth': trial.suggest_categorical('max_depth', US_MODEL_SEARCH_SPACE['max_depth']),
            'gamma': trial.suggest_categorical('gamma', US_MODEL_SEARCH_SPACE['gamma']),
            'min_child_weight': trial.suggest_categorical('min_child_weight', US_MODEL_SEARCH_SPACE['min_child_weight']),
            'max_delta_step': trial.suggest_categorical('max_delta_step', US_MODEL_SEARCH_SPACE['max_delta_step']),
            'subsample': trial.suggest_categorical('subsample', US_MODEL_SEARCH_SPACE['subsample']),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', US_MODEL_SEARCH_SPACE['colsample_bytree']),
            'reg_lambda': trial.suggest_categorical('reg_lambda', US_MODEL_SEARCH_SPACE['reg_lambda']),
            'reg_alpha': trial.suggest_categorical('reg_alpha', US_MODEL_SEARCH_SPACE['reg_alpha'])
        }

        fold_scores = []
        fold_records = []

        for fold, (train_idx, val_idx) in enumerate(make_purged_time_series_splits(X_train_val, CV_SPLITS, N_WINDOW), 1):
            calibration_fit_idx, calibration_idx, evaluation_idx = split_calibration_and_evaluation(val_idx)
            if calibration_fit_idx is None:
                continue

            X_train = X_train_val.iloc[train_idx]
            y_train = y_train_val.iloc[train_idx]
            X_cal_fit = X_train_val.iloc[calibration_fit_idx]
            y_cal_fit = y_train_val.iloc[calibration_fit_idx]
            model = make_model(params)
            model.fit(X_train, y_train, eval_set=[(X_cal_fit, y_cal_fit)], verbose=False)
            calibration_proba = model.predict_proba(X_train_val.iloc[calibration_idx])
            thresh_buy = np.percentile(calibration_proba[:, 1], percentile_threshold)
            thresh_sell = np.percentile(calibration_proba[:, 2], percentile_threshold)
            eval_proba = model.predict_proba(X_train_val.iloc[evaluation_idx])
            eval_prices = prices_train_val.iloc[evaluation_idx].reset_index(drop=True)
            eval_preds = generate_signals_simple(
                eval_proba,
                eval_prices['Close'].values,
                thresh_buy,
                thresh_sell,
                N=N_WINDOW,
                price_percentile_buy=price_percentile_buy,
                price_percentile_sell=price_percentile_sell,
                min_price_gap=min_price_gap
            )

            net_values, trades = run_next_open_backtest(eval_preds, eval_prices)
            metrics = summarize_net_values(net_values)
            if trades.empty:
                fold_scores.append(-1.0)
                continue
            score = metrics['sharpe'] + 0.25 * metrics['annual_return'] + metrics['total_return'] + metrics['max_dd']
            fold_scores.append(score)
            fold_records.append((fold, float(score), safe_best_n_estimators(model)))

        trial.set_user_attr("fold_records", fold_records)
        return float(np.mean(fold_scores))


    print('=====================Optuna starting...========================')
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS)
    print('========================Optuna end.======================')

    conn.execute("DELETE FROM cv_folds WHERE run_id=? AND symbol=?", (run_id, symbol))
    conn.executemany(
        "INSERT OR REPLACE INTO cv_folds VALUES (?,?,?,?,?)",
        [
            (run_id, symbol, fold, score, best_trees)
            # Preserve the sequential pipeline's last-trial cv_folds semantics.
            for fold, score, best_trees in study.trials[-1].user_attrs.get("fold_records", [])
        ],
    )

    best_params = study.best_params.copy()
    best_percentile = best_params.pop('percentile_threshold')
    best_price_buy = best_params.pop('price_percentile_buy')
    best_price_sell = best_params.pop('price_percentile_sell')
    best_gap = best_params.pop('min_price_gap')

    early_start = int(len(X_train_val) * 0.60)
    threshold_start = int(len(X_train_val) * 0.82)
    train_end = early_start - N_WINDOW
    early_end = threshold_start - N_WINDOW
    X_model_train = X_train_val.iloc[:train_end]
    y_model_train = y_train_val.iloc[:train_end]
    X_early_stop = X_train_val.iloc[early_start:early_end]
    y_early_stop = y_train_val.iloc[early_start:early_end]
    X_threshold_cal = X_train_val.iloc[threshold_start:]

    final_model = make_model(best_params)
    final_model.fit(
        X_model_train,
        y_model_train,
        eval_set=[(X_model_train, y_model_train), (X_early_stop, y_early_stop)],
        verbose=False
    )
    final_best_trees = safe_best_n_estimators(final_model)

    cal_proba = final_model.predict_proba(X_threshold_cal)
    final_thresh_buy = np.percentile(cal_proba[:, 1], best_percentile)
    final_thresh_sell = np.percentile(cal_proba[:, 2], best_percentile)
    # The trained model is fixed for the whole holdout window, so one batched
    # prediction is equivalent to 249 one-row calls and avoids repeated
    # CPU-to-GPU DMatrix conversion.
    test_proba = final_model.predict_proba(X_test)
    test_preds = generate_signals_simple(
        test_proba,
        prices_test['Close'].values,
        final_thresh_buy,
        final_thresh_sell,
        N=N_WINDOW,
        price_percentile_buy=best_price_buy,
        price_percentile_sell=best_price_sell,
        min_price_gap=best_gap
    )
    labeled_positions = np.flatnonzero(source_df["label"].notna().to_numpy())
    live_context = source_df.iloc[labeled_positions[-1] + 1:].copy()
    live_prediction_row = None
    if not live_context.empty:
        X_live = live_context[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        live_proba = final_model.predict_proba(X_live)
        combined_proba = np.vstack([test_proba, live_proba])
        combined_close = np.concatenate([
            prices_test["Close"].to_numpy(dtype=float),
            live_context["Close"].to_numpy(dtype=float),
        ])
        combined_preds = generate_signals_simple(
            combined_proba,
            combined_close,
            final_thresh_buy,
            final_thresh_sell,
            N=N_WINDOW,
            price_percentile_buy=best_price_buy,
            price_percentile_sell=best_price_sell,
            min_price_gap=best_gap,
        )
        if not np.array_equal(combined_preds[:len(test_preds)], test_preds):
            raise RuntimeError(f"{symbol}: live context changed historical test signals")
        latest_live = live_context.iloc[-1]
        live_prediction_row = (
            run_id,
            symbol,
            pd.Timestamp(latest_live["Date"]).strftime('%Y-%m-%d'),
            float(latest_live["Open"]),
            float(latest_live["Close"]),
            None,
            int(combined_preds[-1]),
            float(live_proba[-1, 1]),
            float(live_proba[-1, 2]),
            'live',
        )
    net_values, trades = run_next_open_backtest(test_preds, prices_test)
    metrics = summarize_net_values(net_values)
    strict_buy = exact_precision(y_test, test_preds, target_label=1)
    strict_sell = exact_precision(y_test, test_preds, target_label=2)
    tol_buy = tolerance_precision_one_to_one(y_test, test_preds, target_label=1, tol=TOLERANCE)
    tol_sell = tolerance_precision_one_to_one(y_test, test_preds, target_label=2, tol=TOLERANCE)

    model_params = {
        **best_params,
        'percentile_threshold': best_percentile,
        'price_percentile_buy': best_price_buy,
        'price_percentile_sell': best_price_sell,
        'min_price_gap': best_gap,
        'final_buy_probability_threshold': float(final_thresh_buy),
        'final_sell_probability_threshold': float(final_thresh_sell),
    }
    row = (
        run_id, symbol, 'complex_xgb_gpu_us_adjusted',
        model_df.Date.iloc[0].strftime('%Y-%m-%d'),
        model_df.Date.iloc[train_val_end - 1].strftime('%Y-%m-%d'),
        prices_test.Date.iloc[0].strftime('%Y-%m-%d'),
        prices_test.Date.iloc[-1].strftime('%Y-%m-%d'),
        len(X_train_val), len(X_test), json.dumps(model_params, ensure_ascii=False), final_best_trees,
        metrics['total_return'], metrics['annual_return'], metrics['max_dd'], metrics['sharpe'],
        exact_precision(y_test, test_preds, 1), exact_precision(y_test, test_preds, 2),
        tolerance_precision_one_to_one(y_test, test_preds, 1, tol=TOLERANCE),
        tolerance_precision_one_to_one(y_test, test_preds, 2, tol=TOLERANCE),
        int(np.sum(test_preds != 0)), len(trades), market_now_iso()
    )
    conn.execute("INSERT OR REPLACE INTO model_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    pred_rows = [
        (
            run_id, symbol, prices_test.Date.iloc[i].strftime('%Y-%m-%d'),
            float(prices_test.Open.iloc[i]), float(prices_test.Close.iloc[i]), int(y_test[i]),
            int(test_preds[i]), float(test_proba[i, 1]), float(test_proba[i, 2]), 'test'
        )
        for i in range(len(test_preds))
    ]
    conn.executemany("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)", pred_rows)
    if live_prediction_row is not None:
        conn.execute("INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?)", live_prediction_row)
    out_dir = EXPORT_ROOT / symbol.replace('.', '_')
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        'Date': prices_test.Date,
        'Open': prices_test.Open,
        'Close': prices_test.Close,
        '真实拐点编码': y_test,
        '预测信号': test_preds,
        '买入概率': test_proba[:, 1],
        '卖出概率': test_proba[:, 2],
    }).to_csv(out_dir / f'{strip_suffix(symbol)}_测试集预测标签.csv', index=False, encoding='utf-8-sig')
    log.info(
        '%s model done: test=%d signals=%d trades=%d return=%.2f%% live_date=%s live_signal=%s',
        symbol,
        len(test_preds),
        int(np.sum(test_preds != 0)),
        len(trades),
        metrics['total_return'] * 100,
        live_prediction_row[2] if live_prediction_row is not None else None,
        live_prediction_row[6] if live_prediction_row is not None else None,
    )


def compute_features_and_model_worker(
    run_id: str,
    symbol: str,
    raw: pd.DataFrame,
    mode: str,
    model_mode: str,
) -> dict[str, str]:
    """Compute factors and train one symbol in an isolated spawned process."""
    import sqlite3

    conn = sqlite3.connect(DB_PATH, timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    try:
        feats = compute_features(raw)
        labels = compute_labels(raw)
        db_features_labels(conn, run_id, symbol, raw, feats, labels)
        conn.commit()
        if mode == "full":
            if model_mode == "daily_test":
                run_model_daily_test(conn, run_id, symbol, raw, feats, labels)
            elif model_mode == "batch":
                run_model(conn, run_id, symbol, raw, feats, labels)
            else:
                raise ValueError(f"unsupported MODEL_MODE={model_mode!r}")
            conn.commit()
        return {"symbol": symbol, "date": pd.Timestamp(raw["Date"].max()).strftime("%Y-%m-%d")}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run(mode: str, resume_run_id: str | None = None) -> int:
    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key: raise RuntimeError("ALPHAVANTAGE_API_KEY is not set")
    conn = connect_db(); init_db(conn)
    info = gpu_info() if mode == "full" else "GPU check not required for data-only mode"
    if resume_run_id:
        row = conn.execute(
            "SELECT mode,status FROM pipeline_runs WHERE run_id=?",
            (resume_run_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"resume run_id not found: {resume_run_id}")
        if row[0] != mode:
            raise ValueError(
                f"resume mode mismatch for {resume_run_id}: stored={row[0]} requested={mode}"
            )
        run_id = resume_run_id
        conn.execute(
            "UPDATE pipeline_runs SET finished_at=NULL,status='running',error=NULL WHERE run_id=?",
            (run_id,),
        )
        conn.commit()
        log.info("resuming incomplete run=%s previous_status=%s", run_id, row[1])
    else:
        run_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO pipeline_runs(run_id,started_at,status,mode,gpu_info) VALUES (?,?,?,?,?)",
            (run_id, market_now_iso(), "running", mode, info),
        )
        conn.commit()
    log.info("run=%s GPU=%s", run_id, info.replace("\n", "; "))
    log.info(
        "model execution: device=%s workers=%d xgb_n_jobs=%d",
        "gpu" if os.environ.get("REQUIRE_GPU", "1") == "1" else "cpu",
        MODEL_MAX_WORKERS,
        XGB_N_JOBS,
    )
    try:
        refresh_data = os.environ.get("REFRESH_DATA", "1") == "1"
        start_index = max(1, int(os.environ.get("START_INDEX", "1")))
        requested_symbols = {
            item.strip() for item in os.environ.get("SYMBOLS", "").split(",") if item.strip()
        }
        symbols = [symbol for symbol in STOCK_CODES if not requested_symbols or symbol in requested_symbols]
        full_universe = not requested_symbols and start_index == 1
        if resume_run_id and mode == "full" and full_universe:
            model_symbols = {
                row[0] for row in conn.execute(
                    "SELECT DISTINCT symbol FROM model_runs WHERE run_id=?",
                    (run_id,),
                )
            }
            live_dates = {
                row[0]: pd.Timestamp(row[1]) for row in conn.execute(
                    "SELECT symbol,MAX(date) FROM predictions "
                    "WHERE run_id=? AND dataset='live' GROUP BY symbol",
                    (run_id,),
                )
            }
            raw_dates = {
                row[0]: pd.Timestamp(row[1]) for row in conn.execute(
                    "SELECT symbol,MAX(date) FROM raw_daily GROUP BY symbol"
                ) if row[1]
            }
            target_date = max(raw_dates.values()) if raw_dates else pd.Timestamp.min
            symbols = [
                symbol for symbol in STOCK_CODES
                if symbol not in model_symbols
                or symbol not in live_dates
                or symbol not in raw_dates
                or live_dates.get(symbol, pd.Timestamp.min) < target_date
                or raw_dates.get(symbol, pd.Timestamp.min) < target_date
            ]
            log.info(
                "resume coverage: models=%d/%d live=%d/%d target_date=%s; symbols_to_repair=%d: %s",
                len(model_symbols),
                len(STOCK_CODES),
                len(live_dates),
                len(STOCK_CODES),
                target_date.date() if target_date != pd.Timestamp.min else None,
                len(symbols),
                ",".join(symbols) or "none",
            )
        symbols = symbols[start_index - 1:]
        log.info("completed data cutoff (America/New_York, exclusive): %s", completed_data_cutoff().date())
        log.info("symbols scheduled: %d", len(symbols))

        def fetch_symbol(symbol: str) -> pd.DataFrame:
            if refresh_data:
                raw, changed = refresh_raw_incremental(conn, symbol, api_key)
                db_raw(conn, symbol, raw, market_now_iso())
                conn.commit()
                log.info("%s incremental update: changed=%s rows=%d", symbol, changed, len(raw))
            else:
                raw = load_raw_from_db(conn, symbol)
            assert_completed_data(raw, symbol)
            return raw

        pending_data = list(symbols)
        raw_by_symbol: dict[str, pd.DataFrame] = {}
        latest_dates: dict[str, pd.Timestamp] = {}
        data_errors: dict[str, str] = {}
        total_rounds = FAILED_SYMBOL_RETRY_ROUNDS + 1
        for round_index in range(total_rounds):
            if round_index:
                wait = FAILED_SYMBOL_RETRY_BASE_SECONDS * (2 ** (round_index - 1))
                log.warning(
                    "automatic data retry round %d/%d for %d symbols: %s; sleeping %.0fs",
                    round_index,
                    FAILED_SYMBOL_RETRY_ROUNDS,
                    len(pending_data),
                    ",".join(pending_data),
                    wait,
                )
                if wait:
                    time.sleep(wait)

            failed_this_round: list[str] = []
            for index, symbol in enumerate(pending_data, 1):
                log.info(
                    "[data round %d/%d, %d/%d] fetching %s",
                    round_index + 1,
                    total_rounds,
                    index,
                    len(pending_data),
                    symbol,
                )
                try:
                    raw = fetch_symbol(symbol)
                    raw_by_symbol[symbol] = raw
                    latest_dates[symbol] = pd.Timestamp(raw["Date"].max())
                    data_errors.pop(symbol, None)
                except Exception as exc:
                    conn.rollback()
                    safe_error = redact_secret(exc, api_key)
                    data_errors[symbol] = safe_error
                    failed_this_round.append(symbol)
                    log.error("%s data failed in round %d: %s", symbol, round_index + 1, safe_error)
                if refresh_data and index != len(pending_data):
                    time.sleep(DELAY_BETWEEN_REQUESTS)

            # A response can be valid but stale. Compare every successfully read
            # symbol with the newest date seen in this run and retry only laggards.
            stale_symbols: list[str] = []
            if refresh_data and latest_dates:
                target_date = max(latest_dates.values())
                stale_symbols = [
                    symbol for symbol in symbols
                    if symbol in latest_dates and latest_dates[symbol] < target_date
                ]
                for symbol in stale_symbols:
                    data_errors[symbol] = (
                        f"stale data date {latest_dates[symbol].date()} "
                        f"behind run target {target_date.date()}"
                    )
                log.info(
                    "freshness check: target_date=%s current=%d/%d stale=%d failed=%d",
                    target_date.date(),
                    sum(latest_dates.get(symbol) == target_date for symbol in symbols),
                    len(symbols),
                    len(stale_symbols),
                    len(failed_this_round),
                )

            pending_data = list(dict.fromkeys(failed_this_round + stale_symbols))
            if not pending_data:
                break

        data_failures = [
            {"symbol": symbol, "stage": "data", "error": data_errors[symbol]}
            for symbol in pending_data
        ]

        model_failures: list[dict[str, str]] = []
        if mode in ("data", "full"):
            compute_stage = "model" if mode == "full" else "features"
            model_mode = os.environ.get("MODEL_MODE", "daily_test").strip().lower()
            model_pending = [
                symbol for symbol in symbols
                if symbol in raw_by_symbol and symbol not in set(pending_data)
            ]
            model_errors: dict[str, str] = {}
            for model_round in range(total_rounds):
                if not model_pending:
                    break
                if model_round:
                    wait = FAILED_SYMBOL_RETRY_BASE_SECONDS * (2 ** (model_round - 1))
                    log.warning(
                        "automatic model retry round %d/%d for %d symbols: %s; sleeping %.0fs",
                        model_round,
                        FAILED_SYMBOL_RETRY_ROUNDS,
                        len(model_pending),
                        ",".join(model_pending),
                        wait,
                    )
                    if wait:
                        time.sleep(wait)

                worker_count = min(MODEL_MAX_WORKERS, len(model_pending))
                log.info(
                    "parallel %s round %d/%d: symbols=%d workers=%d xgb_n_jobs=%d start_method=spawn",
                    compute_stage,
                    model_round + 1,
                    total_rounds,
                    len(model_pending),
                    worker_count,
                    XGB_N_JOBS,
                )
                failed_models: list[str] = []
                context = mp.get_context("spawn")
                with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
                    futures = {
                        executor.submit(
                            compute_features_and_model_worker,
                            run_id,
                            symbol,
                            raw_by_symbol[symbol],
                            mode,
                            model_mode,
                        ): symbol
                        for symbol in model_pending
                    }
                    for future in as_completed(futures):
                        symbol = futures[future]
                        try:
                            result = future.result()
                            model_errors.pop(symbol, None)
                            log.info("parallel %s completed: %s live_date=%s", compute_stage, symbol, result["date"])
                        except Exception as exc:
                            safe_error = redact_secret(exc, api_key)
                            model_errors[symbol] = safe_error
                            failed_models.append(symbol)
                            log.error(
                                "%s %s failed in parallel round %d: %s",
                                symbol,
                                compute_stage,
                                model_round + 1,
                                safe_error,
                            )
                model_pending = failed_models

            model_failures = [
                {"symbol": symbol, "stage": compute_stage, "error": model_errors[symbol]}
                for symbol in model_pending
            ]

        failures = data_failures + model_failures
        if mode == "full" and full_universe and not failures:
            model_symbols = {
                row[0] for row in conn.execute(
                    "SELECT DISTINCT symbol FROM model_runs WHERE run_id=?",
                    (run_id,),
                )
            }
            live_dates = {
                row[0]: row[1] for row in conn.execute(
                    "SELECT symbol,MAX(date) FROM predictions "
                    "WHERE run_id=? AND dataset='live' GROUP BY symbol",
                    (run_id,),
                )
            }
            raw_dates = {
                row[0]: row[1] for row in conn.execute(
                    "SELECT symbol,MAX(date) FROM raw_daily GROUP BY symbol"
                )
            }
            target_dates = {raw_dates.get(symbol) for symbol in STOCK_CODES}
            coverage_errors = []
            if model_symbols != set(STOCK_CODES):
                coverage_errors.append(
                    f"model coverage {len(model_symbols)}/{len(STOCK_CODES)}"
                )
            if set(live_dates) != set(STOCK_CODES):
                coverage_errors.append(
                    f"live coverage {len(live_dates)}/{len(STOCK_CODES)}"
                )
            if len(target_dates) != 1 or None in target_dates:
                coverage_errors.append(
                    "raw dates are not synchronized: "
                    + ", ".join(str(value) for value in sorted(target_dates, key=str))
                )
            elif set(live_dates.values()) != target_dates:
                coverage_errors.append(
                    "live prediction dates do not match raw target date"
                )
            if coverage_errors:
                failures.append({"symbol": "__coverage__", "error": "; ".join(coverage_errors)})
            else:
                log.info(
                    "publish coverage validated: models=%d live=%d synchronized_date=%s",
                    len(model_symbols),
                    len(live_dates),
                    next(iter(target_dates)),
                )
        status = "partial" if failures else "success"
        error = json.dumps(failures, ensure_ascii=False) if failures else None
        conn.execute("UPDATE pipeline_runs SET finished_at=?,status=?,error=? WHERE run_id=?", (market_now_iso(), status, error, run_id)); conn.commit()
        log.info("pipeline finished: status=%s failures=%d", status, len(failures))
        return 1 if failures else 0
    except Exception as exc:
        log.exception("pipeline failed")
        conn.execute("UPDATE pipeline_runs SET finished_at=?,status=?,error=? WHERE run_id=?", (market_now_iso(), "failed", str(exc)[:2000], run_id)); conn.commit()
        return 1
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["data", "full"], default="full")
    parser.add_argument("--resume-run-id")
    args = parser.parse_args()
    return run(args.mode, args.resume_run_id)


if __name__ == "__main__":
    sys.exit(main())
