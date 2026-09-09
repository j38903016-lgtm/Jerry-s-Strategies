# 美股拐点预测看板

## US Stock Turning-Point Forecast Dashboard

本分支是美股每日模型 pipeline 的轻量公开看板包。它只包含 Streamlit 展示代码和最新完整生产批次的 SQLite 快照，不包含 Alpha Vantage API 密钥、`.env`、训练特征库或服务器信息。

This branch is the lightweight public dashboard package for the US-stock daily model pipeline. It contains only the Streamlit presentation code and the latest complete production snapshot. API keys, environment files, training features, and server details are excluded.

## Streamlit Community Cloud

- Branch: `us-stock-dashboard`
- Entry point: `streamlit_app.py`
- Python dependencies: `requirements.txt`

The server retrains the 65-symbol universe at 06:00 America/New_York on weekdays. After a successful complete run, it exports and pushes a new `data/dashboard.sqlite3`; Streamlit Community Cloud then redeploys from this branch automatically.

## 美股参数调整 / US Parameter Adjustments

相对 A 股源 pipeline，仅调整以下 Optuna 搜索候选或范围；特征、标签、purged CV、评分、早停、回测、手续费、滑点与看板流程不变。

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
| `max_delta_step` | 1, 2 | 1, 2 (unchanged) |
| `subsample` | 0.6, 0.8 | 0.65, 0.8, 0.9 |
| `colsample_bytree` | 0.5, 0.7 | 0.55, 0.7, 0.85 |
| `reg_lambda` | 5, 10 | 5, 8, 12 |
| `reg_alpha` | 1, 3 | 0.5, 1.5, 3 |

## 本地预览 / Local Preview

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

模型信号仅供研究展示，不构成投资建议。

Model signals are for research display only and are not investment advice.
