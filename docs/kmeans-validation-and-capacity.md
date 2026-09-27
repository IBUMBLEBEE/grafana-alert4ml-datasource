# K-means 验证、回放与容量验收手册

本文定义 Funnel 替换交付时必须留下的证据。它区分三类结论：仓库自动化通过、
生产代表性回放通过、生产容量实测通过。前一类不能替代后两类。

## 当前实现的硬边界

以下数值来自当前代码常量或默认配置，不是性能测量结果：

| 边界                    |            当前值 | 实现语义                                             |
| ----------------------- | ----------------: | ---------------------------------------------------- |
| 默认检测步长            |             60 秒 | 固定训练/评分网格，不随 Grafana 缩放                 |
| 默认历史                |           60 小时 | 57 小时模型覆盖加最多 3 小时 anchor 对齐             |
| 特征上下文              |        8 个原始点 | 九个连续原始点产生首个特征                           |
| 上游单分片              |    最多 10,000 点 | 相邻 inclusive 分片共享边界，合并时按时间去重        |
| 单次检测查询原始点      | 最多 1,000,000 点 | 计划阶段按单序列估算，返回后按全部 series/数值列复核 |
| K-means blocking worker |                 2 | 进程内 semaphore；额外任务等待 permit                |
| 上游请求并发            |                 4 | 进程级 semaphore；所有 history/current proxy 共用    |
| 历史内存缓存            | 64 条、TTL 120 秒 | 按查询语义和固定步长区分；只缓存上游 frame           |
| 模型数                  |    最多 18/series | 默认达到 18 个才允许 AlertSignal 给出有效 0/1        |
| 训练向量                |     每窗 24–1,440 | 先构造连续特征，再确定性抽样；有效覆盖率至少 90%     |

历史分片在一次 fetch 内顺序执行；全插件的 history/current proxy 请求共用 4 permit
的进程级 semaphore。等待 permit 的请求仍受 Grafana 请求取消约束。容量测试仍须覆盖
并发 Dashboard/告警请求，以验证目标数据源、超时和排队延迟，而不能只根据 permit
数量推断生产吞吐。

显示聚合不能降低 ML 成本。面板显示 5 分钟点时，`detectIntervalMs: 60000` 仍先按
一分钟完成取数、特征和评分，再聚合显示；`alert_signal` 始终绕过显示聚合。

## 仓库自动化基线

在仓库根目录执行：

```bash
cargo test --manifest-path rsod/Cargo.toml -p rsod-kmeans
cargo test --manifest-path rsod/Cargo.toml -p rsod-backend
cargo test --manifest-path rsod/Cargo.toml --workspace
npm run typecheck
npm run test:ci
npm run build
GRAFANA_URL=http://127.0.0.1:3003 npm run e2e -- \
  tests/kmeans.integration.spec.ts \
  tests/queryEditor.spec.ts \
  tests/alerting.integration.spec.ts \
  --project=chromium --reporter=line
```

每次交付记录提交 SHA、Rust/Node/Grafana 版本、数据库模式、命令、开始/结束时间、
通过数和完整失败原因。不能只记录“通过”。

### 2026-09-27 已有记录

以下是实施计划中已经记录的结果，尚未包含本文要求的生产回放或容量实测：

- `cargo check --workspace`：通过。
- `cargo test -p rsod-kmeans`：23/23 通过（20 unit + 3 Netdata contract integration）。
- `cargo test -p rsod-backend`：66/66 通过。
- `npm run typecheck`：通过。
- `npm run test:ci`：1 suite、8/8 通过。
- `npm run build`：通过。
- Grafana E2E：`kmeans.integration.spec.ts`、`queryEditor.spec.ts`、
  `alerting.integration.spec.ts` 合计 8/8 通过（记录环境为
  `GRAFANA_URL=http://127.0.0.1:3003`，耗时 1.3 分钟）；增加 K-means
  `no_data_points` 断言后定向复测 4/4 通过。
- `cargo test --workspace`：未全绿。既存 `rsod-baseline` daily/weekly 精度测试分别以
  F1=0.2017 和 F1=0.3250 失败；不得把它们记为 K-means 回归，也不得为使工作区变绿
  而放宽断言。

这些计数可能随新增测试变化。发布记录必须重新运行命令，而不是复制本节。

## 标注数据回放协议

### 1. 固定数据集和标签

为每个数据集保存不可变 manifest，至少包含：

```text
dataset_id, source_snapshot, timezone, series_identity,
from, to, detect_interval_ms, raw_point_count,
missing_point_count, label_version, label_owner
```

标签使用事件区间而不是只标一个截图。每个事件至少记录 series identity、开始、结束、
类别、严重度、标签来源和复核人。另设“未知/不可判定”区间，评估时不能把它强行当作
正常。训练、调参和最终验收数据按时间切分，不能随机打散相邻点。

### 2. 构建隔离

分别从基线 `a1e5cbb` 和候选提交构建插件，使用相同 Grafana、上游快照、查询表达式、
时区、评估区间和检测步长。两个构建使用独立数据库/schema 和独立缓存；每轮冷启动
测量前重建隔离环境，不能让候选实现读取另一轮生成的模型或历史缓存。

Funnel 和 K-means 的输出字段不同，比较层统一转换成：

```text
series_identity, timestamp, prediction(0|1|null), score(optional), readiness
```

`null` 不计入 TP/FP/TN/FN，但必须单独报告覆盖率。把 null 当 0 会虚增真阴性，并
掩盖冷启动和数据质量问题。

### 3. 严格因果执行

按事件时间升序回放。评分时刻 `t` 只能使用结束时间不晚于 `t` 且不包含 `t` 的训练
窗口。不得先用整个数据集训练再回算历史。至少执行以下两种轮次：

- 冷启动：空模型库、空历史缓存，从数据集起点开始。
- 热启动：只加载在该评分时刻按因果规则已经生成的模型快照。

同一输入、配置、anchor 和 seed 重跑两次，逐点比较 timestamp、prediction、score、
validity 和 readiness。改变分片大小或缓存冷热状态后再比较一次；有差异即为重现性
失败。

### 4. 指标定义

点级指标只在“有标注且 prediction 非 null”的点上计算：

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2 * precision * recall / (precision + recall)
FPR       = FP / (FP + TN)
coverage  = valid_scored_points / expected_labeled_points
```

同时报告事件级指标：事件命中率、每个正常小时/天的误报事件数、首次有效异常到事件
开始的检测延迟（p50/p95/max）、冷启动到 Ready 的时间。事件命中的容忍窗口必须在
查看候选结果前定义。持续多个点的同一异常不得被当成多个独立事件美化召回率。

结果按指标族、步长、标签基数、缺失率和事件类型分层；宏平均与微平均同时给出。
不得只挑选成功序列，不得把 Netdata 文档中的理论数字当作本项目实测。

### 5. 验收门槛

准确率门槛必须由规则所有者在看到最终验收集结果前填写：

| 指标                         | 门槛         | 实测   | 结论   |
| ---------------------------- | ------------ | ------ | ------ |
| 点级 precision / recall / F1 | 待业务方确定 | 待回放 | 未验收 |
| 事件命中率                   | 待业务方确定 | 待回放 | 未验收 |
| 正常时间误报事件率           | 待业务方确定 | 待回放 | 未验收 |
| 检测延迟 p95                 | 待业务方确定 | 待回放 | 未验收 |
| 有效覆盖率                   | 待业务方确定 | 待回放 | 未验收 |
| 冷启动到 Ready               | 待业务方确定 | 待回放 | 未验收 |

仓库当前没有可证明生产准确率的标注数据和基线/候选 replay 报告，因此 P6 的准确率
验收仍未完成。

## 高基数和长范围容量验收

### 资源估算

对 `S` 条序列、历史长度 `H_ms`、评估长度 `R_ms` 和固定步长 `Δ_ms`，请求前的
近似原始点数为：

```text
points_per_series = ceil(H_ms / Δ_ms) + ceil(R_ms / Δ_ms) + 8
total_raw_points  = S * points_per_series
```

实际返回值还要乘以每个 frame 的数值列数。代码在计划阶段只用
`points_per_series` 与 1,000,000 比较；上游返回后才对全部 frame、series 和数值列
再次检查总量。高基数请求可能已经产生上游流量后才收到 `BudgetExceeded`，因此调用方
需要在发起请求前使用上式估算，不能依赖后端拒绝来做日常限流。

默认 60 小时历史、60 秒步长时，每条序列仅历史部分约 3,600 点，另加评估范围和
8 点上下文。这里是公式计算，不是容量结论；能安全承载多少序列必须以目标环境实测。

### 测试矩阵

每种目标数据源至少覆盖：

- series 数：1、10、50、100，以及预期生产峰值；
- 评估范围：1 小时、24 小时、7 天，以及最大允许范围；
- 固定步长：60 秒、300 秒，以及生产实际值；
- 数据状态：完整、1%/10% 随机缺失、连续缺口、系统性 5 倍降采样；
- 缓存状态：冷缓存、内存命中、TTL 后、Postgres 重启恢复；
- 并发：单面板、多个面板、Dashboard 与告警同时查询；
- 模型状态：空库首次训练、部分模型、18 模型 Ready、重启加载；
- 结果：正常、`WarmingUp`、`ResolutionMismatch`、`BudgetExceeded`、请求取消。

接近 1,000,000 点边界时分别测试边界内、等于边界和超出边界。验证失败是显式错误，
而不是静默增加步长、截断历史或返回部分结果。

### 每轮必须采集

```text
commit/build, Grafana/Rust/Node version, datasource/backend version,
series_count, numeric_columns, from/to, detect_interval_ms,
planned_points, actual_returned_points, upstream_request_count,
chunk_count, cache_hit/miss, model_count/readiness,
wall_time p50/p95/max, CPU time, process RSS peak,
upstream bytes, timeout/cancel count, error type,
model table rows/bytes before and after
```

资源测量使用目标部署的容器/cgroup 指标和上游请求日志。重复至少一次冷缓存轮次和多次
热缓存轮次，报告样本数与波动，不能只取最佳一次。模型快照使用每个 anchor 的不可变
key，容量测试还必须覆盖长时间运行后的数据库增长率和备份/恢复时间。

### 容量结论模板

| 场景     | series | 范围/步长 | raw points | p95 延迟 |  CPU | RSS 峰值 | 上游请求/字节 | DB 增量 | 结论   |
| -------- | -----: | --------- | ---------: | -------: | ---: | -------: | ------------: | ------: | ------ |
| 冷启动   |   待测 | 待测      |       待测 |     待测 | 待测 |     待测 |          待测 |    待测 | 未验收 |
| 热缓存   |   待测 | 待测      |       待测 |     待测 | 待测 |     待测 |          待测 |    待测 | 未验收 |
| 并发峰值 |   待测 | 待测      |       待测 |     待测 | 待测 |     待测 |          待测 |    待测 | 未验收 |
| 预算边界 |   待测 | 待测      |       待测 |     待测 | 待测 |     待测 |          待测 |    待测 | 未验收 |

仓库目前只有预算、分片、缓存合并、worker 和取消的功能测试，没有目标环境的 CPU、
RSS、上游字节、长时间数据库增长及高并发实测。填写本表并由服务所有者签字前，不能
声称 P6 容量验收完成。

## 发布判定

只有以下条件同时满足才可把交付状态标为完成：

1. 定向 Rust、前端和 Grafana E2E 通过；工作区非绿项有归属和不放宽断言的记录。
2. 固定数据集 replay 可重现，候选和基线报告包含误报、漏报、覆盖率、事件延迟和
   冷启动，不只包含截图。
3. 目标环境容量矩阵完成，预算边界、并发、缓存、取消和存储增长均有原始证据。
4. 每个保存的 Funnel 查询都有迁移或保留/回滚决定；告警规则明确 NoData/Error/
   `for` 策略。
5. 迁移和回滚演练通过，旧 JSON、旧插件构建和数据库备份可用。
