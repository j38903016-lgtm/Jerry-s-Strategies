# A股拐点预测看板

## A-Share Turning-Point Forecast Dashboard

该目录是可直接上传到 GitHub 的轻量看板发布包。它只包含 Streamlit 展示代码和最新完整生产批次的看板快照，不包含 Alpha Vantage API 密钥、`.env`、训练特征库或服务器登录信息。

This folder is a GitHub-ready Streamlit dashboard package. It contains only the presentation code and a compact snapshot of the latest complete production run. API keys, `.env`, training features, and server credentials are excluded.

## 部署方式 / Deployment

1. 在 GitHub 新建一个仓库，把本目录中的全部文件上传到仓库根目录。
2. 打开 Streamlit Community Cloud，选择该 GitHub 仓库。
3. 将入口文件设置为 `streamlit_app.py`。
4. 部署完成后使用 Streamlit 提供的固定 HTTPS 地址访问，不再需要 SSH 隧道或 `localhost:8501`。

1. Create a GitHub repository and upload every file in this folder to the repository root.
2. Open Streamlit Community Cloud and select the repository.
3. Set `streamlit_app.py` as the application entry point.
4. Use the permanent HTTPS URL issued by Streamlit after deployment; SSH and `localhost:8501` are no longer required.

GitHub Pages 只能托管静态网页，不能直接运行 Python/Streamlit。本包应通过 GitHub 连接到 Streamlit Community Cloud，而不是直接启用 GitHub Pages。

GitHub Pages hosts static files only and cannot execute Python or Streamlit. Connect this repository to Streamlit Community Cloud instead of enabling GitHub Pages.

## 每日数据同步 / Daily Data Publishing

服务器上的模型 pipeline 仍在北京时间每个工作日 06:00 运行。运行成功后，执行主项目中的 `export_dashboard_snapshot.py`，即可把最新完整批次导出到本仓库的 `data/dashboard.sqlite3`。配置服务器 GitHub deploy key 后，可使用 `publish_dashboard_snapshot.sh` 自动提交并推送更新。

The server pipeline runs at 06:00 Beijing Time on weekdays. After a successful run, `export_dashboard_snapshot.py` exports the latest complete run to `data/dashboard.sqlite3`. Once a GitHub deploy key is configured on the server, `publish_dashboard_snapshot.sh` can commit and push the snapshot automatically.

服务器需要设置：

```bash
export DASHBOARD_REPO_DIR=/home/jerry/a_share_dashboard_repo
/home/jerry/a_share_pipeline/publish_dashboard_snapshot.sh
```

在 GitHub 仓库和部署密钥尚未创建前，请不要把自动推送命令加入 cron。

Do not add the publishing command to cron until the GitHub repository and deploy key have been configured.

## 本地预览 / Local Preview

```bash
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 文件说明 / Package Contents

- `streamlit_app.py`：Streamlit Cloud 入口 / Streamlit Cloud entry point
- `dashboard.py`：双语看板界面 / bilingual dashboard application
- `data/dashboard.sqlite3`：轻量只读看板快照 / compact read-only dashboard snapshot
- `requirements.txt`：看板依赖 / dashboard dependencies
- `.streamlit/config.toml`：主题配置 / theme configuration
- `export_dashboard_snapshot.py`：快照导出工具 / snapshot exporter
- `publish_dashboard_snapshot.sh`：服务器自动推送模板 / server publishing template
