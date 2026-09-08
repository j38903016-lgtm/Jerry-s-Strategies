#!/usr/bin/env python3
"""A screenshot-style, read-only stock-selection monitoring dashboard."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import altair as alt
import pandas as pd
import streamlit as st

SERVER_DB_PATH = Path("/data/a_share_pipeline/a_share_pipeline.sqlite3")
BUNDLED_DB_PATH = Path(__file__).resolve().parent / "data" / "dashboard.sqlite3"
DB_PATH = Path(
    os.environ.get(
        "PIPELINE_DB",
        str(SERVER_DB_PATH if SERVER_DB_PATH.exists() else BUNDLED_DB_PATH),
    )
)
SECTORS = {
    "人工智能": ["688256.SH", "688041.SH", "603019.SH", "601138.SH", "002230.SZ",
            "000977.SZ", "002415.SZ", "688111.SH", "300418.SZ"],
    "机器人": ["300124.SZ", "688017.SH", "002747.SZ", "300024.SZ", "002472.SZ",
            "603728.SH", "601689.SH", "688165.SH", "002050.SZ", "688322.SH"],
    "芯片": ["688041.SH", "688256.SH", "603986.SH", "603986.SH", "603290.SH",
            "300661.SZ", "603501.SH", "300474.SZ", "600745.SH", "301308.SZ"],
    "半导体": ["688981.SH", "002371.SZ", "688012.SH", "600584.SH", "301269.SZ",
            "688126.SH", "688072.SH", "688019.SH", "688347.SH", "002156.SZ"],
    "银行": ["601398.SH", "601939.SH", "601288.SH", "601988.SH", "600036.SH",
            "601658.SH", "601166.SH", "000001.SZ", "002142.SZ", "600926.SH"],
    "黄金": ["601899.SH", "600547.SH", "600489.SH", "600988.SH", "000975.SZ",
            "002237.SZ", "002155.SZ", "601069.SH", "001337.SZ", "600916.SH"],
    "有色金属": ["601899.SH", "600362.SH", "601600.SH", "002460.SZ", "002466.SZ",
            "600111.SH", "603993.SH", "603799.SH", "601168.SH", "600711.SH"],
}
SECTOR_EN = {
    "人工智能": "Artificial Intelligence",
    "机器人": "Robotics",
    "芯片": "Chips",
    "半导体": "Semiconductors",
    "银行": "Banking",
    "黄金": "Gold",
    "有色金属": "Non-ferrous Metals",
}
SYMBOL_SECTOR = {symbol: sector for sector, symbols in SECTORS.items() for symbol in symbols}
EXPECTED_SYMBOL_COUNT = len(SYMBOL_SECTOR)
BEIJING_TZ = ZoneInfo("Asia/Shanghai")

st.set_page_config(
    page_title="A股拐点预测看板 | A-Share Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
<style>
:root {
  --ink: #182539;
  --ink-2: #35445a;
  --muted: #738095;
  --muted-2: #98a2b2;
  --line: #dfe5ed;
  --line-strong: #cfd8e4;
  --panel: #ffffff;
  --canvas: #f4f6f9;
  --buy: #a72f46;
  --buy-bright: #c64059;
  --buy-soft: #fbf1f3;
  --sell: #11705d;
  --sell-bright: #168b72;
  --sell-soft: #edf8f5;
  --amber: #9a6710;
  --blue: #315a82;
}
.stApp {
  background:
    linear-gradient(rgba(37, 55, 79, .018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(37, 55, 79, .018) 1px, transparent 1px),
    var(--canvas);
  background-size: 24px 24px;
  color: var(--ink);
  font-family: Inter, "IBM Plex Sans", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}
.block-container { max-width: 1540px; padding: .55rem 1.45rem 3rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: -.02em; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
.dashboard-hero, .dashboard-card, .signal-card, .meta-item { box-sizing: border-box; }
.dashboard-hero {
  position: relative;
  overflow: hidden;
  margin: 0 0 .65rem;
  padding: 1.05rem 1.2rem .95rem;
  border: 1px solid #203149;
  border-radius: 8px;
  color: #fff;
  background:
    linear-gradient(115deg, rgba(255,255,255,.025) 0 1px, transparent 1px 100%),
    linear-gradient(120deg, #111c2d 0%, #1a2a41 62%, #233950 100%);
  background-size: 18px 18px, auto;
  box-shadow: 0 10px 24px rgba(21, 34, 52, .14);
}
.dashboard-hero::after {
  content: "";
  position: absolute;
  width: 290px;
  height: 290px;
  right: -120px;
  top: -165px;
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 50%;
  box-shadow: 0 0 0 34px rgba(255,255,255,.025), 0 0 0 68px rgba(255,255,255,.018);
}
.hero-top { display: flex; align-items: center; justify-content: space-between; gap: 1rem; position: relative; z-index: 1; }
.hero-kicker { color: #8fa6bf; font-size: 9px; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; }
.hero-title { margin-top: .15rem; color: #fff; font-size: clamp(1.65rem, 2.4vw, 2.25rem); line-height: 1.06; font-weight: 750; letter-spacing: -.035em; }
.hero-title-en { margin-top: .18rem; color: #9fb1c5; font-size: 11px; font-weight: 500; letter-spacing: .04em; }
.hero-status { min-width: 145px; text-align: right; }
.status-pill { display: inline-flex; align-items: center; gap: 7px; padding: 6px 9px; border: 1px solid rgba(126,211,183,.28); border-radius: 4px; color: #a6e1cf; background: rgba(17,112,93,.18); font-size: 9px; font-weight: 800; letter-spacing: .12em; }
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #55d4ac; box-shadow: 0 0 0 4px rgba(85,212,172,.11); }
.status-cn { display: block; margin-top: 5px; color: #7f93aa; font-size: 9px; }
.meta-strip { display: grid; grid-template-columns: 1.35fr .65fr .7fr .95fr 1fr; gap: 0; margin-top: .8rem; border-top: 1px solid rgba(255,255,255,.09); position: relative; z-index: 1; }
.meta-item { min-width: 0; padding: .58rem .8rem 0 0; color: #8fa1b5; font-size: 9px; letter-spacing: .025em; }
.meta-item + .meta-item { padding-left: .8rem; border-left: 1px solid rgba(255,255,255,.08); }
.meta-item strong { display: block; margin-top: 2px; color: #f2f6fa; font-family: "Roboto Mono", "SFMono-Regular", Consolas, monospace; font-size: 11px; font-weight: 650; font-variant-numeric: tabular-nums; }
.meta-item .en, .bi-note .en, .card-en, .heading-en, .signal-en, .legend-en {
  display: block;
  color: var(--muted-2);
  font-size: 8.5px;
  line-height: 1.25;
  font-weight: 500;
  letter-spacing: .035em;
}
.meta-item .en { color: #758aa2; margin-top: 1px; font-size: 7.5px; text-transform: uppercase; }
.section-heading { margin: 0 0 .55rem; padding-left: .6rem; border-left: 3px solid #354f6f; }
.heading-cn { color: var(--ink); font-size: 1.28rem; line-height: 1.05; font-weight: 760; letter-spacing: -.02em; }
.heading-en { margin-top: .16rem; font-size: 8px; letter-spacing: .12em; text-transform: uppercase; }
.bi-note { max-width: 1060px; color: #6f7b8c; font-size: 10px; line-height: 1.45; margin: -.12rem 0 .5rem; }
.bi-note .en { margin-top: .08rem; font-size: 8px; }
.dashboard-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 9px 10px 8px;
  background: var(--panel);
  min-height: 74px;
  box-shadow: 0 2px 7px rgba(31,45,65,.035);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.dashboard-card::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 2px; background: #7890aa; }
.dashboard-card.performance::before { background: var(--buy); }
.dashboard-card.risk::before { background: #bd7741; }
.dashboard-card.quality::before { background: var(--blue); }
.dashboard-card:hover { border-color: var(--line-strong); box-shadow: 0 5px 14px rgba(31,45,65,.07); }
.dashboard-card .title { color: #667388; font-size: 10px; margin-bottom: 8px; line-height: 1.2; }
.dashboard-card .value { color: var(--ink); font-family: "Roboto Mono", "SFMono-Regular", Consolas, monospace; font-size: 17px; font-weight: 750; letter-spacing: -.04em; font-variant-numeric: tabular-nums; }
.dashboard-card.performance .value { color: var(--buy); }
.dashboard-card.risk .value { color: #96592d; }
.dashboard-card.quality .value { color: #274f77; }
.card-en { margin-top: 2px; font-size: 7.5px; text-transform: uppercase; }
.signal-card {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 7px;
  padding: 11px 12px 10px;
  background: #fff;
  min-height: 134px;
  box-shadow: 0 3px 10px rgba(30,45,70,.045);
  transition: border-color .15s ease, box-shadow .15s ease;
}
.signal-card:hover { border-color: var(--line-strong); box-shadow: 0 7px 18px rgba(30,45,70,.075); }
.signal-card.buy { border-top: 3px solid var(--buy); background: linear-gradient(180deg, var(--buy-soft), #fff 48%); }
.signal-card.sell { border-top: 3px solid var(--sell); background: linear-gradient(180deg, var(--sell-soft), #fff 48%); }
.signal-top { display: flex; align-items: center; justify-content: space-between; gap: .5rem; }
.signal-card .symbol { color: var(--ink); font-family: "Roboto Mono", "SFMono-Regular", Consolas, monospace; font-size: 15px; font-weight: 750; letter-spacing: .02em; }
.action-badge { padding: 3px 6px; border-radius: 3px; font-size: 8px; font-weight: 800; letter-spacing: .08em; }
.buy .action-badge { color: var(--buy); background: rgba(167,47,70,.09); }
.sell .action-badge { color: var(--sell); background: rgba(17,112,93,.09); }
.sector-name { margin-top: 8px; color: var(--ink-2); font-size: 10px; font-weight: 650; }
.signal-en { margin-top: 1px; font-size: 7.5px; text-transform: uppercase; }
.confidence-row { display: flex; align-items: end; justify-content: space-between; margin-top: 10px; color: #7c8798; font-size: 8px; }
.confidence-row strong { color: var(--ink); font-family: "Roboto Mono", Consolas, monospace; font-size: 15px; line-height: 1; }
.confidence-row small, .signal-date small { display: block; color: #a0a9b6; font-size: 6.8px; text-transform: uppercase; letter-spacing: .08em; }
.prob-track { height: 3px; margin-top: 5px; overflow: hidden; border-radius: 3px; background: #e9edf2; }
.prob-track span { display: block; height: 100%; border-radius: 3px; }
.buy .prob-track span { background: var(--buy-bright); }
.sell .prob-track span { background: var(--sell-bright); }
.signal-date { display: flex; align-items: end; justify-content: space-between; gap: 8px; margin-top: 9px; padding-top: 7px; border-top: 1px solid #e8ecf1; color: #7d8898; font-size: 8px; }
.signal-date strong { color: #46556a; font-family: "Roboto Mono", Consolas, monospace; font-size: 9px; font-weight: 650; }
.signal-buy { color: var(--buy); font-weight: 800; }
.signal-sell { color: var(--sell); font-weight: 800; }
.stock-grid-header { display: grid; grid-template-columns: 82px 78px 78px 82px 62px 130px; column-gap: 8px; width: 640px; margin: .65rem .45rem .35rem; padding-left: 2.35rem; color: #748196; font-size: 8px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase; }
.stock-grid-header span small { display: block; margin-top: 1px; color: #a0a8b5; font-size: 6.5px; font-weight: 500; }
.chart-legend { text-align: right; color: #5d6d82; font-size: 12px; padding: 4px 4px 0 0; }
.legend-item { display: inline-flex; align-items: center; gap: 5px; margin-left: 15px; }
.legend-en { display: inline; margin-left: 2px; }
.buy-mark { color: var(--buy); font-size: 17px; }
.sell-mark { color: var(--sell); font-size: 17px; }
[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: #d8e0e9 !important;
  border-radius: 8px !important;
  background: rgba(255,255,255,.76);
  box-shadow: 0 4px 14px rgba(34,54,82,.035);
}
[data-testid="stExpander"] {
  border-color: #dde3eb;
  background: rgba(255,255,255,.96);
  border-radius: 6px;
  box-shadow: none;
  margin-bottom: 5px;
}
[data-testid="stExpander"]:hover { border-color: #c8d2de; }
[data-testid="stExpander"] summary { min-height: 42px; }
[data-testid="stExpander"] summary p { color: #34445a; font-family: "Roboto Mono", "Noto Sans SC", Consolas, monospace; font-size: 10.5px; font-weight: 650; line-height: 1.35; font-variant-numeric: tabular-nums; }
[data-testid="stSidebar"] { background: #f7f8fa; border-right: 1px solid #dce2e9; }
[data-testid="stSidebar"] .heading-cn { font-size: 1.05rem; }
[data-testid="stButton"] button { min-height: 34px; border-radius: 5px; border-color: #cbd5e1; color: var(--ink); background: #fff; font-size: 10px; font-weight: 700; }
[data-testid="stButton"] button:hover { border-color: #71859d; color: #203b5a; }
[data-testid="stAlert"] { border-radius: 6px; font-size: 10px; }
@media (max-width: 900px) {
  .block-container { padding: .35rem .65rem 2.5rem; }
  .dashboard-hero { padding: .9rem; }
  .hero-status { display: none; }
  .meta-strip { grid-template-columns: repeat(2, 1fr); }
  .meta-item + .meta-item { padding-left: 0; border-left: 0; }
  .stock-grid-header { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=15)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def card(title_cn: str, title_en: str, value: str, tone: str = "neutral") -> None:
    st.markdown(
        f'<div class="dashboard-card {escape(tone)}"><div class="title">{escape(title_cn)}'
        f'<span class="card-en">{escape(title_en)}</span></div><div class="value">{escape(value)}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(title_cn: str, title_en: str) -> None:
    st.markdown(
        f'<div class="section-heading"><div class="heading-cn">{escape(title_cn)}</div>'
        f'<div class="heading-en">{escape(title_en)}</div></div>',
        unsafe_allow_html=True,
    )


def bilingual_note(text_cn: str, text_en: str) -> None:
    st.markdown(
        f'<div class="bi-note">{escape(text_cn)}<span class="en">{escape(text_en)}</span></div>',
        unsafe_allow_html=True,
    )


def signal_card(symbol: str, sector: str, label: str, probability: float, date: str, kind: str) -> None:
    css = "buy" if "买" in label else "sell"
    label_en = "Buy" if css == "buy" else "Sell"
    sector_en = SECTOR_EN.get(sector, "Other")
    probability_width = min(max(probability * 100, 0.0), 100.0)
    st.markdown(
        f'<div class="signal-card {css}"><div class="signal-top"><div class="symbol">{escape(symbol)}</div>'
        f'<div class="action-badge">{escape(label)} · {label_en.upper()}</div></div>'
        f'<div class="sector-name">{escape(sector)}<span class="signal-en">{escape(sector_en)}</span></div>'
        f'<div class="confidence-row"><span>模型置信度<small>Model Confidence</small></span>'
        f'<strong>{probability:.1%}</strong></div><div class="prob-track"><span style="width:{probability_width:.1f}%"></span></div>'
        f'<div class="signal-date"><span>信号日期<small>Signal Date</small></span><strong>{escape(date)}</strong>'
        f'<span>{escape(kind)}<small>Confirmed</small></span></div></div>',
        unsafe_allow_html=True,
    )


def pct(value) -> str:
    return "—" if pd.isna(value) else f"{float(value) * 100:.2f}%"


def number(value) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2f}"


def latest_run() -> pd.DataFrame:
    return query("SELECT * FROM pipeline_runs ORDER BY julianday(started_at) DESC LIMIT 1")


def latest_publishable_run() -> pd.DataFrame:
    return query(
        """
        SELECT pr.*, mr.model_count, lp.live_count
        FROM pipeline_runs pr
        JOIN (
          SELECT run_id, COUNT(DISTINCT symbol) AS model_count
          FROM model_runs
          GROUP BY run_id
        ) mr ON mr.run_id=pr.run_id
        JOIN (
          SELECT run_id, COUNT(DISTINCT symbol) AS live_count
          FROM predictions
          WHERE dataset='live'
          GROUP BY run_id
        ) lp ON lp.run_id=pr.run_id
        WHERE pr.status='success' AND pr.mode='full'
          AND mr.model_count>=? AND lp.live_count>=?
        ORDER BY julianday(pr.finished_at) DESC
        LIMIT 1
        """,
        (EXPECTED_SYMBOL_COUNT, EXPECTED_SYMBOL_COUNT),
    )


def latest_models(run_id: str | None) -> pd.DataFrame:
    if run_id:
        frame = query("SELECT * FROM model_runs WHERE run_id=? ORDER BY symbol", (run_id,))
        if not frame.empty:
            return frame
    return query(
        """
        WITH ranked AS (
          SELECT m.*, ROW_NUMBER() OVER (
            PARTITION BY symbol ORDER BY julianday(created_at) DESC
          ) AS rank_number
          FROM model_runs m
        )
        SELECT * FROM ranked WHERE rank_number=1 ORDER BY symbol
        """
    )


def all_predictions(run_id: str | None) -> pd.DataFrame:
    if run_id:
        frame = query("SELECT * FROM predictions WHERE run_id=? AND dataset='test' ORDER BY symbol,date", (run_id,))
        if not frame.empty:
            return frame
    return query(
        """
        WITH ranked AS (
          SELECT m.run_id, m.symbol, ROW_NUMBER() OVER (
            PARTITION BY m.symbol ORDER BY julianday(m.created_at) DESC
          ) AS rank_number
          FROM model_runs m
        )
        SELECT p.* FROM predictions p
        JOIN ranked latest ON latest.run_id=p.run_id AND latest.symbol=p.symbol
        WHERE p.dataset='test'
          AND latest.rank_number=1
        ORDER BY p.symbol,p.date
        """
    )


def live_predictions(run_id: str | None) -> pd.DataFrame:
    if not run_id:
        return pd.DataFrame()
    return query(
        "SELECT * FROM predictions WHERE run_id=? AND dataset='live' ORDER BY symbol,date",
        (run_id,),
    )


def build_daily_signals(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ["buy_probability", "sell_probability"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    latest_date = frame["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()
    recent = frame[frame["date"] == latest_date].copy()
    signals = []
    for symbol, group in recent.groupby("symbol"):
        group = group.copy()
        group["predicted_signal"] = pd.to_numeric(group["predicted_signal"], errors="coerce").fillna(0).astype(int)
        group = group[group["predicted_signal"].isin([1, 2])].copy()
        if group.empty:
            continue
        group["kind"] = group.apply(
            lambda row: "买入" if row["predicted_signal"] == 1 else "卖出",
            axis=1,
        )
        group["probability"] = group.apply(
            lambda row: row["buy_probability"] if row["kind"] == "买入" else row["sell_probability"], axis=1
        )
        row = group.iloc[0].copy()
        row["display_label"] = "买点" if row["kind"] == "买入" else "卖点"
        signals.append(row)
    return pd.DataFrame(signals)


def sector_summary(models: pd.DataFrame, daily_signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, symbols in SECTORS.items():
        frame = models[models["symbol"].isin(symbols)].copy()
        if frame.empty:
            continue
        signals = daily_signals[daily_signals["symbol"].isin(symbols)] if not daily_signals.empty else pd.DataFrame()
        rows.append(
            {
                "sector": sector,
                "coverage": len(frame),
                "return": frame["test_total_return"].mean(),
                "annual": frame["test_annual_return"].mean(),
                "drawdown": frame["test_max_drawdown"].mean(),
                "sharpe": frame["test_sharpe"].mean(),
                "buy_count": int((signals["kind"] == "买入").sum()) if not signals.empty else 0,
                "sell_count": int((signals["kind"] == "卖出").sum()) if not signals.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def render_stock_chart(symbol: str, predictions: pd.DataFrame) -> None:
    stock = predictions[predictions["symbol"] == symbol].copy()
    if stock.empty:
        st.info("该品种暂无预测图数据。\n\nNo prediction chart data is available for this symbol.")
        return
    stock["date"] = pd.to_datetime(stock["date"])
    stock["predicted_signal"] = pd.to_numeric(stock["predicted_signal"], errors="coerce").fillna(0).astype(int)
    stock["buy_probability"] = pd.to_numeric(stock["buy_probability"], errors="coerce").fillna(0.0)
    stock["sell_probability"] = pd.to_numeric(stock["sell_probability"], errors="coerce").fillna(0.0)
    signals = stock[stock["predicted_signal"].isin([1, 2])].copy()
    signals["信号"] = signals["predicted_signal"].map({1: "买点", 2: "卖点"})
    signals["形状"] = signals["predicted_signal"].map({1: "triangle-up", 2: "triangle-down"})
    st.markdown(
        '<div class="chart-legend">'
        '<span class="legend-item"><span class="buy-mark">▲</span><span>买点<span class="legend-en">Buy</span></span></span>'
        '<span class="legend-item"><span class="sell-mark">▼</span><span>卖点<span class="legend-en">Sell</span></span></span>'
        '</div>',
        unsafe_allow_html=True,
    )
    price = alt.Chart(stock).mark_line(color="#6b778c", strokeWidth=1.5).encode(
        x=alt.X("date:T", title=["日期", "Date"]),
        y=alt.Y("close:Q", title=["收盘价", "Closing Price"]),
        tooltip=[
            alt.Tooltip("date:T", title="日期 / Date"),
            alt.Tooltip("close:Q", title="收盘价 / Closing Price", format=".2f"),
        ],
    )
    layers = [price]
    if not signals.empty:
        points = alt.Chart(signals).mark_point(size=135, filled=True).encode(
            x="date:T",
            y="close:Q",
            color=alt.Color(
                "信号:N",
                scale=alt.Scale(domain=["买点", "卖点"], range=["#d4667d", "#4f9f83"]),
                legend=None,
            ),
            shape=alt.Shape("形状:N", legend=None),
            tooltip=[
                alt.Tooltip("date:T", title="日期 / Date"),
                alt.Tooltip("close:Q", title="收盘价 / Closing Price", format=".2f"),
                alt.Tooltip("信号:N", title="预测 / Forecast"),
                alt.Tooltip("buy_probability:Q", title="买入概率 / Buy Probability", format=".1%"),
                alt.Tooltip("sell_probability:Q", title="卖出概率 / Sell Probability", format=".1%"),
            ],
        )
        layers.append(points)
    st.altair_chart(alt.layer(*layers).interactive(), width="stretch")


if not DB_PATH.exists():
    st.error(f"数据库不存在：{DB_PATH}\n\nDatabase not found: {DB_PATH}")
    st.stop()

latest_attempt = latest_run()
published_run = latest_publishable_run()
published_run_id = published_run.iloc[0]["run_id"] if not published_run.empty else None
models = latest_models(published_run_id)
predictions = all_predictions(published_run_id)
production_predictions = live_predictions(published_run_id)
daily_signals = build_daily_signals(production_predictions)
sector_df = sector_summary(models, daily_signals)
raw_latest = query("SELECT MAX(date) AS date FROM raw_daily")
raw_date = raw_latest.iloc[0]["date"] if not raw_latest.empty else "—"
prediction_date = production_predictions["date"].max() if not production_predictions.empty else "—"
signal_count = len(daily_signals)

stock_count = models["symbol"].nunique() if not models.empty else 0
generated_at = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M:%S")
st.markdown(
    '<div class="dashboard-hero"><div class="hero-top"><div><div class="hero-kicker">Institutional Quantitative Intelligence</div>'
    '<div class="hero-title">A股拐点预测看板</div>'
    '<div class="hero-title-en">A-Share Turning-Point Forecast Dashboard</div></div>'
    '<div class="hero-status"><div class="status-pill"><span class="status-dot"></span>PRODUCTION ONLINE</div>'
    '<span class="status-cn">生产系统正常 · 数据只读发布</span></div></div>'
    '<div class="meta-strip">'
    f'<div class="meta-item">生成时间 <strong>{generated_at}</strong><span class="en">Generated at · Beijing Time</span></div>'
    f'<div class="meta-item">股票数 <strong>{stock_count}</strong><span class="en">Symbols</span></div>'
    f'<div class="meta-item">有结果 <strong>{stock_count}</strong><span class="en">Completed Results</span></div>'
    f'<div class="meta-item">行情日期 <strong>{escape(str(raw_date))}</strong><span class="en">Market Data Date</span></div>'
    f'<div class="meta-item">生产信号日期 <strong>{escape(str(prediction_date))}</strong><span class="en">Production Signal Date</span></div>'
    '</div></div>',
    unsafe_allow_html=True,
)
if published_run.empty:
    st.warning(
        "尚无覆盖完整股票池的成功生产预测批次；当前仅展示历史测试结果。\n\n"
        "No successful production run covers the full universe; historical test results are shown."
    )
if not latest_attempt.empty and latest_attempt.iloc[0]["status"] != "success":
    st.warning(
        f"最近一次执行状态：{latest_attempt.iloc[0]['status']}；已发布看板不会切换到不完整批次。\n\n"
        f"Latest run status: {latest_attempt.iloc[0]['status']}. The dashboard remains on the last complete run."
    )

with st.sidebar:
    section_title("看板控制", "Dashboard Controls")
    if st.button("刷新 · Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    bilingual_note("数据每 15 秒自动重新读取数据库。", "The database is refreshed automatically every 15 seconds.")
bilingual_note(
    "历史指标来自复杂版模型测试集；每日推荐来自最新完整生产批次的收盘后预测，并按下一交易日开盘执行。",
    "Historical metrics use the complex model test set. Daily recommendations use the latest complete post-close production run and execute at the next market open.",
)

top_left, top_right = st.columns([1.02, 1.38], gap="medium")
with top_left:
    with st.container(border=True):
        section_title("全局指标", "Global Performance")
        global_metrics = [
            ("测试集总收益率", "Test Total Return", models["test_total_return"].mean() if not models.empty else None, True, "performance"),
            ("复合年化收益率", "Compound Annual Return", models["test_annual_return"].mean() if not models.empty else None, True, "performance"),
            ("最大回撤", "Maximum Drawdown", models["test_max_drawdown"].mean() if not models.empty else None, True, "risk"),
            ("夏普比率", "Sharpe Ratio", models["test_sharpe"].mean() if not models.empty else None, False, "neutral"),
            ("严格买入精确率", "Strict Buy Precision", models["strict_buy_precision"].mean() if not models.empty else None, True, "quality"),
            ("严格卖出精确率", "Strict Sell Precision", models["strict_sell_precision"].mean() if not models.empty else None, True, "quality"),
            ("容忍度±2天买入命中率", "Buy Hit Rate (±2 Days)", models["tolerance_buy_precision"].mean() if not models.empty else None, True, "quality"),
            ("容忍度±2天卖出命中率", "Sell Hit Rate (±2 Days)", models["tolerance_sell_precision"].mean() if not models.empty else None, True, "quality"),
        ]
        for start in range(0, len(global_metrics), 4):
            cols = st.columns(4)
            for column, (title_cn, title_en, value, is_pct, tone) in zip(cols, global_metrics[start:start + 4]):
                with column:
                    card(title_cn, title_en, pct(value) if is_pct else number(value), tone)

with top_right:
    with st.container(border=True):
        section_title("今日买卖点推荐", "Today's Buy & Sell Recommendations")
        bilingual_note(
            f"最新完结交易日生产预测的确定信号，共 {signal_count} 条；不展示无信号品种。",
            f"Confirmed signals from the latest completed trading day: {signal_count}. Symbols without signals are hidden.",
        )
        if daily_signals.empty:
            st.info(
                "最新生产预测日没有模型确定买卖点。\n\n"
                "The model produced no confirmed buy or sell signals for the latest production date."
            )
        else:
            signal_columns = min(3, len(daily_signals))
            for start in range(0, len(daily_signals), signal_columns):
                cols = st.columns(signal_columns)
                for column, (_, row) in zip(cols, daily_signals.iloc[start:start + signal_columns].iterrows()):
                    with column:
                        signal_card(
                            row["symbol"],
                            SYMBOL_SECTOR.get(row["symbol"], "其他"),
                            row["display_label"],
                            float(row["probability"]),
                            pd.Timestamp(row["date"]).strftime("%Y-%m-%d"),
                            "模型确认",
                        )

section_title("各板块汇总表现", "Sector Performance Overview")
bilingual_note(
    "板块默认折叠；展开板块后查看该板块全部个股，再展开任意个股查看其买卖点预测图。",
    "Sectors are collapsed by default. Expand a sector to view its stocks, then expand a stock to inspect its predicted turning points.",
)
if sector_df.empty:
    st.info("暂无板块模型结果。\n\nNo sector model results are available.")
else:
    for _, sector_row in sector_df.iterrows():
        sector = sector_row["sector"]
        title_cn = (
            f"{sector}　{int(sector_row['coverage'])}只　|　平均收益 {pct(sector_row['return'])}　|　"
            f"年化 {pct(sector_row['annual'])}　|　回撤 {pct(sector_row['drawdown'])}　|　"
            f"买 {int(sector_row['buy_count'])} / 卖 {int(sector_row['sell_count'])}"
        )
        title_en = (
            f"{SECTOR_EN.get(sector, 'Other')} · N={int(sector_row['coverage'])} · RET {pct(sector_row['return'])} · "
            f"ANN {pct(sector_row['annual'])} · DD {pct(sector_row['drawdown'])} · "
            f"B/S {int(sector_row['buy_count'])}/{int(sector_row['sell_count'])}"
        )
        title = f"**{title_cn}**  \n:gray[{title_en}]"
        with st.expander(title, expanded=False):
            metric_cols = st.columns(5)
            for column, (label_cn, label_en, value, tone) in zip(
                metric_cols,
                [
                    ("板块平均收益", "Sector Average Return", pct(sector_row["return"]), "performance"),
                    ("板块平均年化", "Sector Annual Return", pct(sector_row["annual"]), "performance"),
                    ("板块平均回撤", "Sector Average Drawdown", pct(sector_row["drawdown"]), "risk"),
                    ("板块平均夏普", "Sector Average Sharpe", number(sector_row["sharpe"]), "neutral"),
                    ("模型买/卖", "Model Buy / Sell", f"{int(sector_row['buy_count'])} / {int(sector_row['sell_count'])}", "quality"),
                ],
            ):
                with column:
                    card(label_cn, label_en, value, tone)
            sector_models = models[models["symbol"].isin(SECTORS[sector])].copy().sort_values("test_total_return", ascending=False)
            sector_signals = daily_signals[daily_signals["symbol"].isin(SECTORS[sector])] if not daily_signals.empty else pd.DataFrame()
            st.markdown(
                '<div class="stock-grid-header"><span>证券代码<small>Symbol</small></span>'
                '<span>测试收益<small>Test Return</small></span><span>复合年化<small>Annual</small></span>'
                '<span>最大回撤<small>Drawdown</small></span><span>夏普比率<small>Sharpe</small></span>'
                '<span>当前信号<small>Signal</small></span></div>',
                unsafe_allow_html=True,
            )
            for _, stock_row in sector_models.iterrows():
                symbol = stock_row["symbol"]
                stock_signal = sector_signals[sector_signals["symbol"] == symbol] if not sector_signals.empty else pd.DataFrame()
                suffix = "—"
                if not stock_signal.empty:
                    signal_en = "BUY" if stock_signal.iloc[0]["kind"] == "买入" else "SELL"
                    suffix = f"{stock_signal.iloc[0]['display_label']}/{signal_en} {float(stock_signal.iloc[0]['probability']):.1%}"
                stock_title = (
                    f"`{symbol}`　│　{pct(stock_row['test_total_return'])}　│　{pct(stock_row['test_annual_return'])}　│　"
                    f"{pct(stock_row['test_max_drawdown'])}　│　{number(stock_row['test_sharpe'])}　│　{suffix}"
                )
                with st.expander(stock_title, expanded=False):
                    detail_cols = st.columns(5)
                    details = [
                        ("测试集收益", "Test Return", pct(stock_row["test_total_return"]), "performance"),
                        ("复合年化", "Compound Annual Return", pct(stock_row["test_annual_return"]), "performance"),
                        ("最大回撤", "Maximum Drawdown", pct(stock_row["test_max_drawdown"]), "risk"),
                        ("夏普比率", "Sharpe Ratio", number(stock_row["test_sharpe"]), "neutral"),
                        ("模型信号数", "Model Signal Count", str(int(stock_row["signal_count"])), "quality"),
                    ]
                    for column, (label_cn, label_en, value, tone) in zip(detail_cols, details):
                        with column:
                            card(label_cn, label_en, value, tone)
                    section_title(f"{symbol} 买卖点预测图", f"{symbol} Predicted Buy & Sell Points")
                    render_stock_chart(symbol, predictions)

with st.expander("**数据库运行记录**  \n:gray[Database Run History]", expanded=False):
    st.dataframe(query("SELECT * FROM pipeline_runs ORDER BY julianday(started_at) DESC LIMIT 50"), width="stretch", hide_index=True)
