# US Stock Daily Model Pipeline

<sub>美股每日模型 Pipeline 完整源码与生产运行说明。</sub>

This directory contains the complete production source used to fetch US daily market data, compute factors and turning-point labels, train per-symbol models, write synchronized predictions to SQLite, export the public dashboard snapshot, and publish it to GitHub.

<sub>本目录包含生产环境实际使用的完整代码：行情获取、因子与拐点标签、逐股训练、SQLite 写入、看板快照导出和 GitHub 发布。</sub>

## Design invariants

- Raw data, features, labels, model metadata, backtest predictions, and live signals are stored by symbol and `run_id`.
- Only completed `America/New_York` trading dates are accepted.
- Time-series CV is purged by the label window to reduce leakage.
- A failed or stale symbol is retried without rerunning symbols that already completed correctly.
- A public snapshot requires 203/203 model and live-prediction coverage on one synchronized date.
- Model execution uses 16 processes and 4 XGBoost threads per process by default.

<sub>数据库按股票及 `run_id` 保存结果；仅接收美东已完结交易日；时间序列交叉验证按标签窗口隔离；失败补跑只处理缺失或滞后股票；203 只全部完成且日期一致后才允许发布。</sub>

## Universe

The production universe contains 16 groups with 13 slots each: 208 sector slots and 203 unique symbols after preserving intentional cross-group overlap. All original 65 symbols remain; 138 symbols were added on 2026-09-22.

The original thematic groups are Artificial Intelligence, Robotics, Chips, Semiconductors, Banking, Gold, and Non-ferrous Metals. The coverage expansion adds Health Care, Consumer Staples, Utilities, Real Estate, Energy, Consumer Discretionary, Communication Services, Industrials & Defense, and Insurance.

The structure uses the MSCI/S&P GICS sector framework as a coverage backbone and balances offensive, cyclical, and defensive groups. It improves diversification but does not claim strict market-beta or factor neutrality.

<sub>当前标的池为 16 个板块、每板块 13 个席位，共 208 个板块席位、203 只去重股票。原 65 只全部保留，新增 138 只；该结构改善攻守覆盖，但不代表严格市场中性。</sub>

New symbols are explicitly maintained in `ADDED_US_SECTORS_20260922`; `pipeline.py` validates the expected 16-sector/203-symbol configuration at import time.

<sub>新增股票集中维护在 `ADDED_US_SECTORS_20260922`，代码启动时会校验 16 板块和 203 只股票。</sub>

## US-specific model search ranges

These are the model-search changes relative to the original A-share source. Features, labels, scoring, purged CV, early stopping, transaction costs, slippage, and execution rules remain unchanged.

| Parameter | A-share source | US pipeline |
|---|---:|---:|
| `percentile_threshold` | 80–96 | 82–97 |
| `price_percentile_buy` | 5–25 | 5–30 |
| `price_percentile_sell` | 75–95 | 70–95 |
| `min_price_gap` | 0.005–0.05 | 0.0075–0.06 |
| `n_estimators` | 600–1200 | 650–1300 |
| `learning_rate` | 0.01, 0.02, 0.03 | 0.01, 0.015, 0.02, 0.03 |
| `max_depth` | 2, 3 | 2, 3, 4 |
| `gamma` | 0.5, 1, 2 | 0.5, 1, 1.5, 2 |
| `min_child_weight` | 10, 20 | 8, 12, 20 |
| `max_delta_step` | 1, 2 | 1, 2 |
| `subsample` | 0.6, 0.8 | 0.65, 0.8, 0.9 |
| `colsample_bytree` | 0.5, 0.7 | 0.55, 0.7, 0.85 |
| `reg_lambda` | 5, 10 | 5, 8, 12 |
| `reg_alpha` | 1, 3 | 0.5, 1.5, 3 |

<sub>以上为相对 A 股源代码的全部美股模型搜索范围调整；其他核心建模和回测逻辑保持不变。</sub>

## Installation

```bash
cd pipeline
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
# Set ALPHAVANTAGE_API_KEY and review all paths in .env.
./run_pipeline.sh
```

<sub>复制 `.env.example` 后填写 Alpha Vantage Key，并根据本机环境调整数据库、导出和发布目录。</sub>

Useful commands:

```bash
# Data and features only
.venv/bin/python pipeline.py --mode data

# Full pipeline for selected symbols
SYMBOLS=MSFT,NVDA ./run_pipeline.sh

# Resume only missing or stale work in an existing run
./resume_pipeline.sh RUN_ID

# Install the production scheduler
./install_cron.sh

# Start the local read-only dashboard
./run_dashboard.sh
```

## Production schedule and recovery

The production host uses an `Asia/Shanghai` system clock, so cron calls the guard at both possible Shanghai equivalents of 06:00 New York. `run_pipeline_at_6_et.sh` accepts exactly one trigger at 06:00 `America/New_York`, automatically covering EDT and EST.

Each Alpha Vantage request has bounded retries. After the first pass, only failed or stale symbols are retried. If the batch remains incomplete, `run_pipeline.sh` resumes the same `run_id`; it does not retrain symbols already complete for the synchronized target date.

<sub>生产服务器通过双候选 cron 加美东时间守卫覆盖夏令时；行情和模型失败均采用定向补跑，同一批次中正确完成的股票不会重复训练。</sub>

## Publishing

`export_dashboard_snapshot.py` copies only the latest complete production run and the raw rows required by the dashboard. `publish_dashboard_snapshot.sh` commits the snapshot and dashboard code, then updates both `main` and `us-stock-dashboard` so the primary repository and the existing Streamlit deployment remain synchronized.

<sub>快照只导出最新完整批次；发布脚本同时更新 `main` 与 `us-stock-dashboard`，保证主分支和现有 Streamlit 部署一致。</sub>

## Files intentionally excluded from Git

- `.env` and all secrets
- Main SQLite databases and Optuna/training state
- Logs, exports, backups, PID files, temporary files, and virtual environments
- Historical one-off migration scripts that are not part of the current production path

<sub>正式环境变量、主数据库、训练状态、日志、导出、备份、PID、临时文件、虚拟环境及已停用的一次性迁移脚本不会上传。</sub>

## Disclaimer

This software is for research and monitoring only and does not constitute investment advice.

<sub>本软件仅供研究和监控，不构成投资建议。</sub>
