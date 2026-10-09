# 方法设计、公式与理论依据

## 1. QuantLab 到底有没有“最优持有期”

核验基于 QuantLab `main` 的提交 `c6a15a184194d93fe21a09a5e8bbc00c8dae818b`。

结论是：**有多期限诊断工具，没有自动持有期选择器**。

- [因子分析文档](https://github.com/ZhaorongDai/quantlab/blob/main/docs/zh-CN/factor.md) 支持同时传入 1、5、20 等多个前瞻收益期限，生成 `ic_decay_table()`、IC—期限曲线和 95% Newey–West 区间。文档措辞是这些图“提示合适的持有期”。
- [factor_report.py](https://github.com/ZhaorongDai/quantlab/blob/main/quantlab/analysis/factor_report.py) 实现了 `newey_west_lags`、`newey_west_t_stat` 和 `ic_decay_table`，但没有对 horizon 执行 `argmax`、交叉验证、数据窥探修正或保存“最优周期”。
- [组合构建文档](https://github.com/ZhaorongDai/quantlab/blob/main/docs/zh-CN/portfolio.md) 只是建议令 `rebalance_periods` 等于预测标签的 span，使优化期限和实际持仓期限一致；两者均由用户预先设置。

因此，本 sidecar 迁移了 QuantLab 的两个可靠思想：多期限衰减曲线和重叠收益的 Newey–West 修正；逐股选择、板块收缩、最终留出集和执行状态机则是新增部分。

## 2. 数据合同与不干扰原则

研究样本只来自最新完整生产 run 的：

```sql
predictions.dataset = 'test' AND predictions.predicted_signal = 1
```

当日建议来自同一个生产 run 的：

```sql
predictions.dataset = 'live' AND predictions.predicted_signal = 1
```

价格只读查询 `raw_daily`。选择最新 run 时要求 `pipeline_runs.status='success'`、`mode='full'`，并且 `model_runs` 和 live predictions 覆盖配置中的 203 个标的。

数据库连接使用：

```text
file:/data/us_stock_pipeline/us_stock_pipeline.sqlite3?mode=ro
PRAGMA query_only=ON
```

sidecar 的结果写入另一数据库，不存在任何写回生产表的代码路径。

## 3. 固定持有期回测

股票 (i) 在信号日 (t) 出现买点，候选持有期为 (h\in\mathcal H)。和原 pipeline 一致，订单在下一交易日开盘成交。令 (O_{i,t}) 为开盘价、(s) 为滑点、(c_b,c_s) 为买卖费率：

\[
P^{\mathrm{in}}_{i,t,h}=O_{i,t+1}(1+s)(1+c_b)
\]

\[
P^{\mathrm{out}}_{i,t,h}=O_{i,t+1+h}(1-s)(1-c_s)
\]

\[
R_{i,t,h}=\frac{P^{\mathrm{out}}_{i,t,h}}{P^{\mathrm{in}}_{i,t,h}}-1.
\]

默认参数直接复制原 pipeline：(c_b=0.00031)、(c_s=0.00081)、(s=0.001)。同一股票尚未退出时出现的新买点会被忽略，禁止加仓和重叠持仓。原卖点不参与任何计算。

回测不是简单对交易收益求均值；它按每日收盘盯市，空仓日收益为 0，因此 Sharpe 同时反映资金闲置时间。普通年化 Sharpe 为：

\[
\widehat{SR}_{i,h}=\sqrt{252}\frac{\bar r_{i,h}}{s(r_{i,h})}.
\]

固定持有会制造序列相关。延续 QuantLab 和 [Newey–West](https://www.nber.org/papers/t0055) 的方法，长期方差估计为：

\[
\widehat\Omega=\widehat\gamma_0+2\sum_{\ell=1}^{L}
\left(1-\frac{\ell}{L+1}\right)\widehat\gamma_\ell,
\]

\[
L=\max\left(h-1,\left\lfloor4(T/100)^{2/9}\right\rfloor\right),
\qquad
\widehat{SR}^{NW}_{i,h}=\sqrt{252}\frac{\bar r_{i,h}}{\sqrt{\widehat\Omega}}.
\]

[Lo 的 Sharpe 统计研究](https://alo.mit.edu/publications/page/18/)说明忽略序列相关会显著扭曲 Sharpe 的年化与排序。

## 4. 为什么原测试集仍要再次切分

原 XGBoost 的 test 对预测模型是样本外，但一旦拿它来挑 (h)，它就变成了**新模型的开发集**。如果在同一段数据上挑最高 Sharpe 再报告这个最高 Sharpe，就发生第二层过拟合。

因此按共同交易日期顺序切分：

- 前 60%：开发训练段；
- 中间 20%：开发验证段；
- 最后 20%：最终审计段；
- 任一交易若退出日跨越分界线，则从该段剔除，避免前一段偷看后一段价格。

最终审计段从不参与选周期。其理论动机与 [Cawley & Talbot 关于模型选择过拟合的研究](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)一致；金融回测的多重尝试风险还可参见 [Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf)、[Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) 和 [White Reality Check](https://onlinelibrary.wiley.com/doi/pdf/10.1111%2F1468-0262.00152)。

当前只有约 252 个共同测试日，不能再做很深的嵌套滚动 CV；60/20/20 是可运行的最低限度，不代表证据已经充分。积累 12–24 个月 point-in-time live 买点后，应升级为 expanding walk-forward，并报告 PBO/DSR。

## 5. 每只股票一个周期，但用板块层级收缩

对每只股票、每个候选 (h)，先对训练段和验证段的 NW Sharpe 求开发期平均 (S_{i,h})。同一板块股票的 (S_{i,h}) 以 \(\sqrt{n}\) 为权重聚合成板块曲线 \(\bar S_{g,h}\)。单股分数向板块分数收缩：

\[
\eta_{i,h}=\frac{n_{i,h}}{n_{i,h}+\kappa},\qquad \kappa=8,
\]

\[
S^{\mathrm{shrunk}}_{i,h}=\eta_{i,h}S_{i,h}+
(1-\eta_{i,h})\frac{1}{|G_i|}\sum_{g\in G_i}\bar S_{g,h}.
\]

股票属于多个主题板块时，板块先验取其所有板块曲线的平均；用于仓位约束的“主板块”则固定为配置中的第一个板块，避免重复计算权重。最终：

\[
h_i^*=\arg\max_{h\in\mathcal H}S^{\mathrm{shrunk}}_{i,h},
\]

完全相同的最高分取较短周期，保证结果唯一、确定、可复现。层级收缩的统计依据可参见 [Efron–Morris 的经验贝叶斯收缩](https://statistics.stanford.edu/technical-reports/steins-estimation-rule-and-its-competitors-empirical-bayes-approach)。

“唯一”表示每只股票只有一个当前有效周期，不表示 203 只股票必须取 203 个不同数字。

## 6. 置信度与上线门槛

- 高：开发段成熟交易至少 20 笔，最终审计段至少 5 笔；
- 中：开发段成熟交易至少 8 笔；
- 低：其余全部。

当前数据绝大多数是低置信度。建议真正替代人工卖点前至少满足：

1. 大部分拟交易标的达到中等置信度；
2. 至少 6 个月的每日 point-in-time 影子结果；
3. chosen horizon 在连续多次窗口更新中稳定；
4. 最终审计和影子交易均未显著劣于等权/固定 5 日基准；
5. 对所有候选期限计算 DSR 或 White Reality Check，而不是只看最大 Sharpe。

## 7. 仓位权重

只对当日 live 买点形成候选集。历史预期收益取所选周期在开发段的平均日收益年化，并按持有期置信度缩放；`buy_probability` 只做截面倾斜，不解释为真实成功概率：

\[
\hat\mu_i=252\bar r_i\,q_i+0.03z(p_i),
\]

其中 (q_i\in\{0.6,0.8,1.0\}) 分别对应低、中、高置信度，(z(p_i)) 是当日买入概率的标准分。

风险矩阵使用过去 252 个交易日的开盘收益，并用 [Ledoit–Wolf shrinkage](https://ledoit.net/honey.pdf) 估计 \(\widehat\Sigma_{LW}\)。权重求解：

\[
\min_{w\ge0}\quad \frac{\lambda}{2}w^\top\widehat\Sigma_{LW}w-\hat\mu^\top w,
\qquad \lambda=6,
\]

约束为：

\[
0\le w_i\le15\%,\qquad
\sum_{i\in g}w_i\le35\%,\qquad
\sum_iw_i\le100\%.
\]

可行总仓位由当日标的数量和板块集中度决定，剩余部分明确作为现金，不强迫满仓。目标来自 [Markowitz 均值—方差理论](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1952.tb01525.x)，协方差收缩用于抑制估计误差；[DeMiguel、Garlappi、Uppal](https://doi.org/10.1093/rfs/hhm075) 对优化组合样本外不稳定性的证据意味着看板必须同时保留等权/逆波动基准，后续版本应补上该比较。

## 8. 当前不能忽略的两项工程风险

### 8.1 原价格表不是完整复权 OHLC

生产代码请求 `TIME_SERIES_DAILY_ADJUSTED`，但数据库只保存原始 open/high/low/close/volume，没有 adjusted close、dividend 和 split coefficient。拆股会被当成巨大亏损或盈利。sidecar 会告警超过 40% 的开盘跳变，并在风险协方差中 winsorize，但这不能替代复权。

在“不改原 pipeline”的约束下，最稳妥的下一步是让 sidecar 自己建立独立 corporate-action 表并重建复权 OHLC；不得悄悄修改原 `raw_daily`。

### 8.2 原 pipeline 有时错过次日开盘

信号规则假定下一交易日开盘成交，但 2026-10-09 的生产 run 到 09:38 ET 才完成。sidecar 检测到这种情况会把拟入场顺延至再下一个交易日，并显示 `NEXT_OPEN_MISSED`。这使真实执行与历史回测产生一天偏差，后续影子比较必须单列。

