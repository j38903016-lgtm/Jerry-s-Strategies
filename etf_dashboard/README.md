# A股 ETF 拐点预测看板

## A-Share ETF Turning-Point Forecast Dashboard

该目录是 ETF 策略的 GitHub/Streamlit Community Cloud 轻量发布包，只包含展示代码和最新完整生产批次快照，不包含 Alpha Vantage API 密钥、`.env`、训练特征库或服务器登录信息。

入口文件：`streamlit_app.py`

数据文件：`data/dashboard.sqlite3`

服务器模型管线在每个工作日北京时间 06:00 运行；完整成功后，发布脚本自动更新本目录快照。每个代码均为可独立交易的场内 ETF，未使用成分股代替 ETF。

This is the GitHub/Streamlit Community Cloud package for the A-share ETF strategy. It contains only the dashboard code and a compact snapshot of the latest complete production run. API keys, server credentials, raw training data, and `.env` files are excluded.
