# 美股持有期与仓位优化 Sidecar

这是部署在 `/home/jerry/us_stock_pipeline/holding_optimizer` 的独立研究/影子运行模块。它只以 SQLite `mode=ro` 和 `PRAGMA query_only=ON` 查询原 pipeline，不导入 `pipeline.py`，不写原数据库，不改原 cron，不连接券商，也不会生成真实订单。

## 已实现

- 从最新一个覆盖 203 只股票的成功 `full` run 读取 `dataset='test'` 的预测结果；
- 只使用 `predicted_signal=1` 的买点，完全忽略原卖点；
- 每只股票分别在 1、2、3、5、7、10、15、20、30、40 个交易日中选择一个确定持有期；
- 按时间把原测试集再分为 60% 开发训练、20% 开发验证、20% 最终审计，跨边界但尚未退出的交易一律剔除；
- 用板块层级收缩缓解单股买点太少的问题；
- 对最新 `dataset='live'` 买点生成固定退出日期和仓位建议；
- 用 Ledoit–Wolf 协方差和带个股/板块上限、可保留现金的均值—方差优化分配权重；
- 独立 SQLite、独立锁、独立日志和独立 Streamlit 看板（服务器本地 `127.0.0.1:8504`）。

数学、数据切分、偏差控制及论文依据见 [METHODOLOGY.md](METHODOLOGY.md)。首次实盘库影子运行结果见 [RUN_REPORT.md](RUN_REPORT.md)。

## 运行

服务器复用父目录已有的 Python runtime，但不安装包、不改父环境：

```bash
cd /home/jerry/us_stock_pipeline/holding_optimizer
./run_optimizer.sh
./run_dashboard.sh
```

独立数据库：

```text
/data/us_stock_pipeline/holding_optimizer/optimizer.sqlite3
```

本地开发：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt pytest
PYTHONPATH=. .venv/bin/pytest -q
```

`cron.example` 只是模板，部署过程没有把它安装进 crontab。只有影子验证通过并得到授权后，才应增加独立调度；即使增加调度，也不应调用或修改原 pipeline。

## Dashboard 发布边界

`streamlit_app.py` 可直接部署到 Streamlit Community Cloud。服务器已经启动独立的内网看板，但真正创建新的 `*.streamlit.app` 应用需要 Streamlit Cloud 账户会话；代码不能用 GitHub SSH 推送权限替代这一步。公开快照可用：

```bash
python export_dashboard_snapshot.py
```

公开仓库只应包含 `streamlit_app.py`、`optimizer/dashboard.py`、`optimizer/__init__.py`、依赖文件和 `data/optimizer.sqlite3`，不得包含生产数据库或 `.env`。

