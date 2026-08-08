# Alert4ML 接口规范总览

> **版本**: 0.2.0 · **状态**: Draft · **更新**: 2026-08-08
>
> v0.2.0：Go 后端已由 `rsod/crates/rsod-backend`（grafana-plugin-sdk-rust）替换，
> 层间调用不再经过 CGO/Arrow FFI。历史 FFI 协议见
> [go-rust-interface.md](go-rust-interface.md)（已废弃，仅作追溯）。

## 架构分层

```
┌─────────────────────────────────────┐
│  TypeScript (前端)                   │  src/types.ts
│  Alert4MLQuery (JSON)                │  src/datasource.ts
└──────────────┬──────────────────────┘
               │ Grafana Plugin SDK (gRPC)
               │ /api/ds/query 代理上游数据源
               ▼
┌─────────────────────────────────────┐
│  Rust (插件后端)                     │  rsod/crates/rsod-backend/src/
│  Alert4MLQueryJson + HyperParams     │  contract.rs / pipeline.rs
└──────────────┬──────────────────────┘
               │ 直接函数调用（无跨语言边界）
               ▼
┌─────────────────────────────────────┐
│  Rust (ML 引擎)                      │  rsod/crates/rsod-{outlier,baseline,forecaster,funnel}
│  算法 crate + Options struct         │  rsod_core::{DetectionResult, TimeSeriesInput}
└─────────────────────────────────────┘
```

## 核心枚举（全局共享）

### supportDetect × detectType 合法组合

| `supportDetect` | `detectType` | 状态 | rsod 入口（rsod-backend） |
|----------------|-------------|------|--------------------------|
| `baseline` | `dynamics` | ✅ 可用 | `rsod_baseline::dynamics::dynamics_detect` |
| `machine_learning` | `outlier` | ✅ 可用 | `rsod_outlier::outlier` |
| `machine_learning` | `forecast` | ✅ 可用 | `rsod_forecaster::forecast` |
| `machine_learning` | `funnel` | ✅ 可用 | `rsod_funnel::funnel_detect`（双查询） |
| `machine_learning` | `changepoint` | 🔒 保留 | — |

> 其他组合在 rsod-backend `parse_hyper_params()` 阶段返回 error。

## 时间戳单位约定

| 层 | 字段 | 单位 | 类型 |
|----|------|------|------|
| TS | `historyTimeRange.from/to` | 相对秒（距当前） | `number` |
| rsod-backend | `HistoryTimeRange.durationMs` | 毫秒 | `int64` |
| rsod-backend | 上游 frame col[0] | Arrow `Timestamp(ns/ms)`（或数值秒） | `i64` / `f64` |
| rsod-backend → 算法 | `TimeSeriesInput.timestamps` | Unix 秒 | `f64` |
| 算法 → 结果帧 | `DetectionResult.timestamps` | Unix 毫秒 | `i64` → `DateTime` |

> `frame_ops::field_time_ns` 负责统一上游时间列的读取（Timestamp ns/ms、
> Float64/Int64 按 Unix 秒解释），渲染层输出 `DateTime` 列。

## 默认值注入层

每个字段的默认值由且仅由一层注入，避免双重覆盖。

| 字段 | 注入层 | 来源 |
|------|--------|------|
| `historyTimeRange` | TS | `DEFAULT_TIME_RANGE = {from:300, to:0}` |
| `hyperParams` 初始值 | TS | `DEFAULT_RSOD_PARAMS` / `DEFAULT_DYNAMICS_PARAMS` / `DEFAULT_FORECAST_PARAMS` |
| `uniqueKeys` | TS | Grafana 模板变量 `${__dashboard.uid}` + `panelId` + `refId` |
| HyperParams 空字段兜底 | rsod-backend | `contract.rs` 中各 struct 的 `impl Default`（镜像 Go `SetDefaults()`） |
| 算法 Options 字段 | rsod 算法 crate | `serde default` / `impl Default` |

## AI 使用指引

本规范设计为机器可读，遵守以下约定：

1. **Schema 优先**：每份文档包含字段约束表（required/optional、nullable、默认值、注入层、枚举范围）。
2. **示例驱动**：每种 `detectType` 提供最小有效请求样例和完整请求样例。
3. **派生规则显式化**：所有运行时计算（时间范围重算、UUID v5 派生、`targets` 注入）均以编号步骤列出。
4. **跨层映射明确**：每个字段标注 TS 字段名与 Rust serde 字段名，不依赖读者自行推断命名转换。
5. **错误语义完整**：每个接口边界均标注失败时的返回形式和错误传播方式。

## 变更规则

| 变更类型 | 兼容性 | 要求 |
|---------|--------|------|
| 新增 optional 字段 | ✅ 向后兼容 | 无需版本升级，更新对应规范文档 |
| 字段改名 | ❌ Breaking | 两个主版本内保留别名，先 deprecate 再移除 |
| 新增 `detectType` | ✅ 向后兼容 | 同时更新 README 组合矩阵与 `parse_hyper_params` |
| 修改结果帧 schema 列数/类型 | ❌ Breaking | 需要升级主版本 |

## 层间职责

| 层 | 职责 | 不做什么 |
|---|---|---|
| TypeScript | 查询编辑 UI、模板变量替换、参数组装 | 不做 ML 计算 |
| Rust（rsod-backend） | 查询解析、上游数据源代理、帧切分、结果帧渲染、存储初始化 | 不做 ML 算法逻辑 |
| Rust（算法 crate） | ML 算法执行（异常检测、预测、基线、funnel） | 不感知 Grafana 概念 |

## 关键约定

1. TS→后端序列化格式：JSON，字段名 camelCase
2. 后端→算法：直接函数调用（`TimeSeriesInput` + Options），无跨语言边界
3. 结果帧：SDK `Frame`（arrow2 数组），列名沿用 Go 时代的 `Time`/`Anomaly`/`Baseline`/`Pred`/`lower_bound`/`upper_bound`
4. 模型 key：UUID v5，`unique_keys_uuid` + `derive_uuid` 与 Go 字节级兼容（见 `rsod-backend/src/uuid_util.rs`）
5. 错误处理：TS→后端走 Grafana SDK 标准错误；单查询失败返回 per-query `QueryError::Internal`
