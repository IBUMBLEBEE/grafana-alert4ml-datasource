# Alert4ML 接口规范总览

> **版本**: 0.1.0 · **状态**: Draft · **更新**: 2026-04-20

## 架构分层

```
┌─────────────────────────────────────┐
│  TypeScript (前端)                   │  src/types.ts
│  Alert4MLQuery (JSON)                │  src/datasource.ts
└──────────────┬──────────────────────┘
               │ Grafana Plugin SDK
               │ POST /api/ds/query
               ▼
┌─────────────────────────────────────┐
│  Go (后端)                           │  pkg/plugin/types.go
│  Alert4MLQueryJson + HyperParams     │  pkg/plugin/datasource.go
└──────────────┬──────────────────────┘
               │ CGO · Arrow C Data Interface
               │ + JSON options string
               ▼
┌─────────────────────────────────────┐
│  Rust (ML 引擎)                      │  rsod/crates/rsod-ffi/src/lib.rs
│  FFI 函数 + Options struct            │  rsod/crates/rsod-{outlier,baseline,forecaster}
└─────────────────────────────────────┘
```

## 核心枚举（全局共享）

### supportDetect × detectType 合法组合

| `supportDetect` | `detectType` | 状态 | Rust FFI 函数 |
|----------------|-------------|------|--------------|
| `baseline` | `dynamics` | ✅ 可用 | `dynamics_fit_predict` |
| `machine_learning` | `outlier` | ✅ 可用 | `outlier_fit_predict` |
| `machine_learning` | `forecast` | ✅ 可用 | `rsod_forecaster` |
| `machine_learning` | `changepoint` | 🔒 保留 | — |

> 其他组合在 Go `ParseHyperParams()` 阶段返回 error，不会到达 Rust。

## 时间戳单位约定

| 层 | 字段 | 单位 | 类型 |
|----|------|------|------|
| TS | `historyTimeRange.from/to` | 相对秒（距当前） | `number` |
| Go | `HistoryTimeRange.From/To` | 相对秒 | `uint` |
| Go | `Alert4MLQueryBody.From/To` | 绝对时间 | `time.Time` |
| Go → Rust | Arrow col[0] (input) | Unix 毫秒（f64） | `float64` |
| Rust output | `outlier_fit_predict` time | Unix 毫秒（f64） | `float64` |
| Rust output | `baseline_fit_predict` / `dynamics_fit_predict` timestamp | Unix 秒（i64） | `int64` |
| Rust output | `rsod_forecaster` timestamp | Unix 毫秒（f64） | `float64` |

> ⚠️ baseline/dynamics 返回 `i64`（Unix 秒），与 outlier/forecast 的 `f64` 不同。Go 渲染层负责统一转换为 `time.Time`。

## 默认值注入层

每个字段的默认值由且仅由一层注入，避免双重覆盖。

| 字段 | 注入层 | 来源 |
|------|--------|------|
| `historyTimeRange` | TS | `DEFAULT_TIME_RANGE = {from:300, to:0}` |
| `hyperParams` 初始值 | TS | `DEFAULT_RSOD_PARAMS` / `DEFAULT_DYNAMICS_PARAMS` / `DEFAULT_FORECAST_PARAMS` |
| `uniqueKeys` | TS | Grafana 模板变量 `${__dashboard.uid}` + `panelId` + `refId` |
| `RsodHyperParams` 空字段兜底 | Go | `SetDefaults()` |
| `DynamicsHyperParams` 空字段兜底 | Go | `SetDefaults()` |
| `ForecastHyperParams` 空字段兜底 | Go | `SetDefaults()` |
| Rust Options 字段 | Rust | `serde default` / `impl Default` |

## AI 使用指引

本规范设计为机器可读，遵守以下约定：

1. **Schema 优先**：每份文档包含字段约束表（required/optional、nullable、默认值、注入层、枚举范围）。
2. **示例驱动**：每种 `detectType` 提供最小有效请求样例和完整请求样例。
3. **派生规则显式化**：所有运行时计算（时间范围重算、UUID 派生、`targets` 注入）均以编号步骤列出。
4. **跨层映射明确**：每个字段标注 TS 字段名、Go JSON tag、Rust serde 字段名，不依赖读者自行推断命名转换。
5. **错误语义完整**：每个接口边界均标注失败时的返回形式和错误传播方式。

## 变更规则

| 变更类型 | 兼容性 | 要求 |
|---------|--------|------|
| 新增 optional 字段 | ✅ 向后兼容 | 无需版本升级，更新对应规范文档 |
| 字段改名 | ❌ Breaking | 两个主版本内保留别名，先 deprecate 再移除 |
| 新增 `detectType` | ✅ 向后兼容 | 同时更新 README 组合矩阵、ts-go、go-rust 三份文档 |
| 修改 Arrow output schema 列数/类型 | ❌ Breaking | 需要升级主版本
```

## 层间职责

| 层 | 职责 | 不做什么 |
|---|---|---|
| TypeScript | 查询编辑 UI、模板变量替换、参数组装 | 不做 ML 计算 |
| Go | 查询解析、嵌套数据源代理、Arrow 转换、DataFrame 渲染 | 不做 ML 算法逻辑 |
| Rust | ML 算法执行（异常检测、预测、基线） | 不感知 Grafana 概念 |

## 关键约定

1. TS→Go 序列化格式：JSON，字段名 camelCase
2. Go→Rust 数据传输：Apache Arrow C Data Interface（零拷贝）
3. Go→Rust 配置传输：JSON 字符串，字段名 snake_case
4. 时间戳：TS 层用相对秒数，Go 层用 `time.Time`，Rust 层用 `Float64`（Unix 秒）
5. 错误处理：TS→Go 走 Grafana SDK 标准错误，Go→Rust 仅 bool 返回值
