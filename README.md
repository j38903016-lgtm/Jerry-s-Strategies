# 美股拐点预测看板

## US Stock Turning-Point Forecast Dashboard

本分支是美股每日模型 pipeline 的轻量公开看板包。它只包含 Streamlit 展示代码和最新完整生产批次的 SQLite 快照，不包含 Alpha Vantage API 密钥、`.env`、训练特征库或服务器信息。

This branch is the lightweight public dashboard package for the US-stock daily model pipeline. It contains only the Streamlit presentation code and the latest complete production snapshot. API keys, environment files, training features, and server details are excluded.

## Streamlit Community Cloud

- Branch: `us-stock-dashboard`
- Entry point: `streamlit_app.py`
- Python dependencies: `requirements.txt`

The server retrains the 65-symbol universe at 06:00 America/New_York on weekdays. After a successful complete run, it exports and pushes a new `data/dashboard.sqlite3`; Streamlit Community Cloud then redeploys from this branch automatically.

## 本地预览 / Local Preview

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

模型信号仅供研究展示，不构成投资建议。

Model signals are for research display only and are not investment advice.
