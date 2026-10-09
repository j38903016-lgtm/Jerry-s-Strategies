from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st


def _read(conn: sqlite3.Connection, sql: str, params=()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def render(database: str | Path) -> None:
    st.set_page_config(page_title="美股持有期与仓位优化", page_icon="📐", layout="wide")
    st.title("美股持有期与仓位权重 Sidecar")
    path = Path(database)
    if not path.exists():
        st.warning(f"尚无独立模型数据库：{path}")
        return
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    run = _read(conn, "SELECT * FROM optimizer_runs WHERE status='success' ORDER BY finished_at DESC LIMIT 1")
    if run.empty:
        st.warning("尚无成功运行。")
        return
    run_id = run.iloc[0]["optimizer_run_id"]
    recs = _read(conn, "SELECT * FROM daily_recommendations WHERE optimizer_run_id=? ORDER BY recommended_weight DESC", (run_id,))
    portfolio = _read(conn, "SELECT * FROM portfolio_summaries WHERE optimizer_run_id=?", (run_id,))
    selections = _read(conn, "SELECT * FROM horizon_selections WHERE optimizer_run_id=? ORDER BY primary_sector,symbol", (run_id,))
    curves = _read(conn, "SELECT * FROM horizon_results WHERE optimizer_run_id=?", (run_id,))
    warnings = _read(conn, "SELECT * FROM audit_warnings WHERE optimizer_run_id=? ORDER BY severity DESC", (run_id,))

    st.caption(
        f"独立模型运行 {run_id[:10]} · 原 pipeline 运行 {run.iloc[0]['source_run_id'][:10]} · "
        f"更新 {run.iloc[0]['finished_at']}"
    )
    a, b, c, d = st.columns(4)
    a.metric("当日买点", len(recs))
    b.metric("已分配仓位", f"{float(portfolio.iloc[0]['invested_weight']):.1%}" if not portfolio.empty else "—")
    c.metric("现金保留", f"{float(portfolio.iloc[0]['cash_weight']):.1%}" if not portfolio.empty else "—")
    d.metric("低置信度周期", int((selections["confidence"] == "low").sum()))

    st.subheader("今日结论")
    if recs.empty:
        st.info("最新完整生产批次没有买点，因此不生成新买入权重。")
    else:
        shown = recs[[
            "symbol", "primary_sector", "buy_probability", "chosen_horizon", "confidence",
            "proposed_entry_date", "proposed_exit_date", "exchange_calendar_exact",
            "recommended_weight", "risk_contribution",
        ]].copy()
        shown["exchange_calendar_exact"] = shown["exchange_calendar_exact"].map(
            {1: "XNYS", 0: "工作日近似"}
        )
        shown.columns = [
            "股票", "主板块", "买入概率", "持有交易日", "置信度", "拟入场", "拟退出",
            "交易日历", "建议权重", "风险贡献",
        ]
        st.dataframe(
            shown.style.format({"买入概率": "{:.2%}", "建议权重": "{:.2%}", "风险贡献": "{:.2%}"}),
            width="stretch",
        )
        st.caption("当日买点仓位分配")
        st.bar_chart(shown.set_index("股票")[["建议权重"]])

    tabs = st.tabs(["逐股持有期", "板块全景", "样本外审计", "运行告警"])
    with tabs[0]:
        symbol = st.selectbox("股票", selections["symbol"].tolist())
        selected = selections[selections["symbol"] == symbol].iloc[0]
        cols = st.columns(5)
        cols[0].metric("最终持有期", f"{int(selected['chosen_horizon'])} 个交易日")
        cols[1].metric("开发段交易", int(selected["dev_trades"]))
        cols[2].metric("样本外交易", int(selected["test_trades"]))
        cols[3].metric("样本外 NW Sharpe", f"{selected['test_nw_sharpe']:.2f}")
        cols[4].metric("个股信息权重", f"{selected['shrinkage_weight']:.1%}")
        chart = curves[curves["symbol"] == symbol].copy()
        st.caption(f"{symbol}：持有期—Newey–West 修正 Sharpe 曲线（红色结论见上方指标）")
        st.line_chart(chart.pivot(index="horizon", columns="split", values="nw_sharpe"))
        st.dataframe(
            chart[["horizon", "split", "n_trades", "nw_sharpe", "total_return", "max_drawdown"]],
            width="stretch",
            hide_index=True,
        )
    with tabs[1]:
        sector = st.selectbox("板块", sorted(selections["primary_sector"].unique()), key="sector")
        sector_frame = selections[selections["primary_sector"] == sector]
        st.caption(f"{sector}：逐股唯一持有期")
        st.bar_chart(sector_frame.sort_values("chosen_horizon").set_index("symbol")[["chosen_horizon"]])
        st.dataframe(sector_frame, width="stretch")
    with tabs[2]:
        st.caption("开发段选择分数与最终样本外表现")
        st.scatter_chart(
            selections,
            x="selection_score",
            y="test_nw_sharpe",
            color="confidence",
            size="dev_trades",
        )
        st.caption("最终样本外段从未参与持有期选择；它只用于审计，不回填寻优。")
    with tabs[3]:
        st.dataframe(warnings, width="stretch", hide_index=True)
        st.code("生产库连接：SQLite mode=ro + PRAGMA query_only=ON\n不写生产表，不下单，不导入 pipeline.py")
    conn.close()
