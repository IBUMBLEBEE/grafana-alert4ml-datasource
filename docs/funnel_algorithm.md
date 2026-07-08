# Funnel 漏斗型时序异常检测算法

本文件描述 `rsod-funnel` 的整体流程与 L1 / L2 算法流程，用于说明一段时序数据在漏斗架构中的处理路径。

## 1. 设计目标

Funnel 采用漏斗型分层检测：

- **L1（统计预筛）**：偏统计的轻量算法，对绝大多数点做 O(1) 快速裁决，直接判定明显正常和明显异常的点。
- **L2（复杂算法）**：只对 L1 无法判定的灰区点（`Uncertain`）升级到较重的 ML 算法（Outlier / Forecast / Baseline）。

目标是让大部分正常数据在 L1 被过滤，只有少量不确定的点进入 L2，兼顾精度与成本。

对应实现：

- L1：[rsod/crates/rsod-funnel/src/l1.rs](../rsod/crates/rsod-funnel/src/l1.rs)
- Profile：[rsod/crates/rsod-funnel/src/profile.rs](../rsod/crates/rsod-funnel/src/profile.rs)
- Pipeline：[rsod/crates/rsod-funnel/src/pipeline.rs](../rsod/crates/rsod-funnel/src/pipeline.rs)
- L2 路由：[rsod/crates/rsod-funnel/src/l2.rs](../rsod/crates/rsod-funnel/src/l2.rs)
- 度量指标：[rsod/crates/rsod-funnel/src/metrics.rs](../rsod/crates/rsod-funnel/src/metrics.rs)

## 2. 整体流程图

```mermaid
flowchart TD
    A["输入时序数据<br/>current: (timestamp, value)<br/>history: 历史窗口"] --> B["构建/加载 SeasonalProfile"]

    B --> B1["趋势识别<br/>Daily / Weekly / Monthly / None"]
    B1 --> B2["按季节桶聚合历史样本<br/>bucket by time-of-day / weekday / month"]
    B2 --> B3["剔除历史异常样本<br/>Hampel filter"]
    B3 --> B4["生成 L1 统计画像<br/>median + MAD / std"]

    B4 --> C["L1 轻量统计过滤<br/>逐点 O(1) 判定"]

    C --> D{"点是否落入统计区间？"}

    D -->|"inner band 内"| E["Normal<br/>直接判正常<br/>不进入 L2"]
    D -->|"outer band 外"| F["Anomaly<br/>直接判异常<br/>不进入 L2"]
    D -->|"inner 和 outer 之间"| G["Uncertain<br/>进入 L2 复杂算法"]

    G --> H["序列分类器<br/>rsod-classifier"]
    H --> I{"序列类型 / 决策引擎 decide()"}

    I -->|"Stationary"| J["L2: Outlier<br/>rsod-outlier / EIF"]
    I -->|"Trending"| K["L2: Forecast<br/>rsod-forecaster"]
    I -->|"Seasonal / SeasonalWithTrend"| L["L2: Baseline / Forecast<br/>按 decision 路由"]
    I -->|"Irregular"| M["L2: Baseline 或 Outlier<br/>按 confidence / skewness 路由"]

    J --> N["L2 输出异常结果"]
    K --> N
    L --> N
    M --> N

    E --> O["结果合并<br/>merge_l1_l2"]
    F --> O
    N --> O

    O --> P["Grafana 输出<br/>baseline / lower_bound / upper_bound / anomaly"]
```

## 3. L1 算法流程

L1 基于历史构建的季节画像，对当前每个点做 O(1) 三态判定，不做重训练。

```mermaid
flowchart TD
    A["历史数据 history"] --> B["按周期建桶<br/>Daily / Weekly / Monthly"]
    B --> C["每个 bucket 计算鲁棒统计量"]
    C --> D["baseline = median"]
    C --> E["scale = 1.4826 × MAD"]

    D --> F["生成阈值带"]
    E --> F

    F --> G["inner band<br/>baseline ± k_inner × scale"]
    F --> H["outer band<br/>baseline ± k_outer × scale"]

    I["当前点 x"] --> J{"x 的位置"}

    J -->|"inner band 内"| K["Normal"]
    J -->|"outer band 外"| L["Anomaly"]
    J -->|"两者之间"| M["Uncertain → L2"]
```

L1 的三态判定语义（[FilterVerdict](../rsod/crates/rsod-funnel/src/l1.rs)）：

| 判定 | 条件 | 处理 |
| --- | --- | --- |
| `Normal` | 落在内带 `inner band` 内 | 直接判正常，不进入 L2 |
| `Anomaly` | 落在外带 `outer band` 外 | 直接判异常，不进入 L2 |
| `Uncertain` | 落在内带与外带之间 | 升级到 L2 复杂算法 |

阈值方法（MAD / IQR / ZScore）由历史数据的偏度自动选择。

## 4. L2 算法流程

L2 不是固定单一模型，而是先对序列分类，再由决策引擎路由到具体算法，且**只覆盖 L1 判为 `Uncertain` 的点**。

```mermaid
flowchart TD
    A["L1 存在 Uncertain 点"] --> B["对 history 或 current 做序列分类<br/>rsod-classifier::classify"]
    B --> C["SeriesCharacteristic + confidence + skewness"]

    C --> D{"decide()"}

    D -->|"Stationary"| E["Outlier Detector<br/>rsod-outlier / EIF"]
    D -->|"Trending"| F["Forecaster<br/>rsod-forecaster"]
    D -->|"Seasonal / SeasonalWithTrend"| G["Baseline / Forecast<br/>根据周期与趋势"]
    D -->|"Irregular"| H["Outlier / Baseline<br/>根据 confidence 和 skewness"]

    E --> I["L2 检测结果 DetectionResult"]
    F --> I
    G --> I
    H --> I

    I --> J["只覆盖 L1=Uncertain 的点<br/>merge_l1_l2"]
    J --> K["与 L1 Normal / Anomaly 合并输出"]
```

一句话概括：

> L1 用历史统计画像把明显正常和明显异常的点快速裁掉；只有落在灰区的 `Uncertain` 点才触发 L2。L2 先判断时序类型，再路由到 Outlier、Forecast 或 Baseline 算法，最后只覆盖这些灰区点的判定结果。
>
> 注意：L2 当前在 Grafana 面板路径中默认关闭（`enable_l2 = false`，见 [pkg/plugin/datasource.go](../pkg/plugin/datasource.go)），可在算法侧显式开启。

## 5. 一段时序经过 L1/L2 的示例

假设当前窗口有 10 个点：

| 点位 | 原始值 | L1 判定 | 是否进入 L2 | 最终结果 |
| --- | ---: | --- | --- | --- |
| t1 | 100 | Normal | 否 | 正常 |
| t2 | 102 | Normal | 否 | 正常 |
| t3 | 98 | Normal | 否 | 正常 |
| t4 | 135 | Uncertain | 是 | 由 L2 决定 |
| t5 | 160 | Anomaly | 否 | 异常 |
| t6 | 101 | Normal | 否 | 正常 |
| t7 | 129 | Uncertain | 是 | 由 L2 决定 |
| t8 | 99 | Normal | 否 | 正常 |
| t9 | 97 | Normal | 否 | 正常 |
| t10 | 180 | Anomaly | 否 | 异常 |

对应的漏斗分流：

```mermaid
flowchart LR
    A["10 个输入点"] --> B["L1 统计过滤"]

    B --> C["7 个 Normal<br/>直接放行"]
    B --> D["2 个 Anomaly<br/>直接报警"]
    B --> E["2 个 Uncertain<br/>进入 L2"]

    E --> F["L2 复杂算法<br/>Classifier + Outlier / Forecast / Baseline"]
    F --> G["确认或否决边界异常"]

    C --> H["最终输出"]
    D --> H
    G --> H
```

## 6. 可度量的漏斗效果

`funnel_detect_with_metrics` 在检测的同时返回 [FunnelMetrics](../rsod/crates/rsod-funnel/src/metrics.rs)，用于量化漏斗分流：

| 指标 | 含义 |
| --- | --- |
| `total_points` | 当前 eval 窗口点数 |
| `l1_normal` / `l1_anomaly` / `l1_uncertain` | L1 三态分流数量 |
| `l1_coverage_rate` | L1 直接裁决比例 `(normal + anomaly) / total` |
| `l2_escalation_rate` | 升级到 L2 的比例 `uncertain / total` |
| `l2_enabled` / `l2_triggered` | L2 是否开启 / 本次是否触发 |
| `l2_method` | 本次 L2 实际路由到的算法 |
| `l1_elapsed_ms` / `l2_elapsed_ms` / `total_elapsed_ms` | 各阶段耗时 |

漏斗目标可表述为：`l1_coverage_rate` 尽量高（大部分点在 L1 裁决），`l2_escalation_rate` 保持在较低区间。

## 7. 可视化与基准

- 面板可视化示例：[rsod/crates/rsod-funnel/examples/funnel_viz.rs](../rsod/crates/rsod-funnel/examples/funnel_viz.rs)
- 指标基准（含 `L1_cov` / `L2_rate` / 耗时）：[rsod/crates/rsod-funnel/examples/funnel_bench.rs](../rsod/crates/rsod-funnel/examples/funnel_bench.rs)

运行方式：

```bash
cd rsod
cargo run --example funnel_viz -p rsod-funnel --release
cargo run --example funnel_bench -p rsod-funnel
```

可视化输出目录：`dataset/output/funnel_viz/`。
