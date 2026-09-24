# US Stock Turning-Point Forecast Pipeline

<sub>美股拐点预测 Pipeline 与公开看板（当前仓库主策略）。</sub>

This repository's `main` branch contains the primary US-equity strategy, its public Streamlit dashboard, the latest complete dashboard snapshot, and the complete reproducible pipeline source under [`pipeline/`](./pipeline/).

<sub>`main` 分支现以美股策略为主，包含公开看板、最新完整快照，以及位于 `pipeline/` 目录中的完整可复现源码。</sub>

The previous A-share root strategy is preserved on the [`a-share-archive-20260924`](../../tree/a-share-archive-20260924) branch. The [`us-stock-dashboard`](../../tree/us-stock-dashboard) branch remains synchronized for compatibility with the existing Streamlit deployment.

<sub>原 A 股根策略已归档至 `a-share-archive-20260924`；`us-stock-dashboard` 分支继续同步，以兼容现有 Streamlit 部署。</sub>

## Strategy at a glance

- **Universe:** 16 balanced sector groups, 13 slots per group, 203 unique US-listed symbols.
- **Data:** Alpha Vantage daily adjusted OHLCV, using only completed New York trading sessions.
- **Signals:** turning-point labels, engineered price/volume factors, purged time-series validation, and Optuna-tuned XGBoost.
- **Execution assumption:** signals are generated after the close for the next market open, with fees and slippage included in backtests.
- **Production schedule:** 06:00 `America/New_York` on weekdays, safely handling daylight-saving changes.
- **Parallelism:** 16 symbol workers × 4 XGBoost threads per model.
- **Publishing:** a snapshot is released only when all 203 symbols have synchronized raw-data and live-prediction dates.

<sub>策略覆盖 16 个攻守均衡板块、203 只去重股票；使用已完结美股交易日数据，在收盘后生成下一开盘信号。生产训练采用 16 进程 × 每模型 4 线程，只有 203 只全部完整且日期一致时才发布。</sub>

## Repository layout

| Path | Purpose |
|---|---|
| `streamlit_app.py` | Streamlit Community Cloud entry point |
| `dashboard.py` | Read-only bilingual US-stock dashboard |
| `data/dashboard.sqlite3` | Compact snapshot of the latest complete production run |
| `pipeline/pipeline.py` | Full data, feature, label, model, backtest, and persistence pipeline |
| `pipeline/run_pipeline.sh` | Locked production runner with targeted automatic recovery |
| `pipeline/run_pipeline_at_6_et.sh` | Daylight-saving-safe 06:00 ET scheduler guard |
| `pipeline/resume_pipeline.sh` | Explicit same-`run_id` targeted resume |
| `pipeline/export_dashboard_snapshot.py` | Complete-run snapshot exporter |
| `pipeline/publish_dashboard_snapshot.sh` | Dual-branch GitHub publisher |
| `pipeline/.env.example` | Safe configuration template; contains no credentials |

<sub>根目录用于公网 Streamlit 展示；`pipeline/` 保存完整训练、补跑、调度、快照和发布代码。正式 `.env`、API Key、主数据库、日志、虚拟环境及训练导出不会进入 Git。</sub>

## Run the dashboard locally

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

The bundled database is read-only and contains only the latest publishable dashboard data. See [`pipeline/README.md`](./pipeline/README.md) for full pipeline setup and production operation.

<sub>内置数据库仅供只读展示；完整 pipeline 安装、定时运行和补跑方法见 `pipeline/README.md`。</sub>

## Security and reproducibility

Credentials are read from environment variables. The repository intentionally excludes `.env`, SSH keys, raw training databases, logs, exports, backups, PID files, and virtual environments. `.env.example` uses placeholders only.

<sub>凭据仅通过环境变量读取；仓库明确排除正式 `.env`、SSH 密钥、训练数据库、日志、导出、备份、PID 和虚拟环境。</sub>

## Research disclaimer

This project is for quantitative research and monitoring only. Model signals are not investment advice, and sector-count balance is not the same as strict beta, factor, or dollar neutrality.

<sub>本项目仅用于量化研究与监控，不构成投资建议；板块数量均衡不等于严格的 beta、因子或美元敞口中性。</sub>
