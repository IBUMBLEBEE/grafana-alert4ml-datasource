# Alert4ML Go-Rust 交互接口规范

> 依据当前实现整理，来源于 [pkg/rsod/rsod.go](pkg/rsod/rsod.go)、[rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 和 [rsod/crates/rsod-ffi/include/rsod_go.h](rsod/crates/rsod-ffi/include/rsod_go.h)。

## 1. 适用范围

本文件描述 Alert4ML 在 Go 后端与 Rust ML 引擎之间的进程内交互协议，覆盖以下内容：

- Go 如何把 Grafana `data.Frame` 转为 Arrow 并传入 Rust
- Rust FFI 暴露了哪些 C ABI 入口
- 每个入口函数的输入输出 schema、JSON options 和返回语义
- 当前实现下的内存所有权、错误语义和已知不一致点

本文件只描述 Go-Rust 边界，不描述 TS-Go 查询协议。

## 2. 总体交互模型

Go 与 Rust 的交互统一遵循两条通道：

1. 时序数据通过 Apache Arrow C Data Interface 传输
2. 算法参数通过 JSON C 字符串传输

运行时调用链如下：

```text
Grafana data.Frame
  -> Go: data.FrameToArrowTable
  -> Go: cdata.ExportArrowRecordBatch
  -> C ABI: FFI_ArrowSchema* + FFI_ArrowArray* + const char* options_json
  -> Rust FFI: import_ffi_struct_array / parse_json_options
  -> Rust 算法 crate
  -> Rust FFI: export_ffi_result
  -> Go: cdata.ImportCRecordBatch / data.FromArrowRecord
```

其中：

- Go 侧入口位于 [pkg/rsod/rsod.go](pkg/rsod/rsod.go)
- Rust FFI 边界位于 [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs)
- C 头文件位于 [rsod/crates/rsod-ffi/include/rsod_go.h](rsod/crates/rsod-ffi/include/rsod_go.h)

## 3. C ABI 总览

当前头文件暴露 5 个入口：

```c
bool outlier_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *_history_schema,
    FFI_ArrowArray *_history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool baseline_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool dynamics_fit_predict(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool rsod_forecaster(
    FFI_ArrowSchema *data_schema,
    FFI_ArrowArray *data_array,
    FFI_ArrowSchema *history_schema,
    FFI_ArrowArray *history_array,
    const char *_options_json,
    FFI_ArrowSchema *result_schema,
    FFI_ArrowArray *result_array);

bool rsod_storage_init(bool trial_mode, const char *pg_dsn);
```

除 `rsod_storage_init` 外，其余 4 个入口都遵循相同的参数顺序：

```text
data_schema, data_array, history_schema, history_array, options_json, result_schema, result_array
```

其中 `outlier_fit_predict` 为了保持接口统一，仍保留 history 参数位，但当前实现不会读取历史数据。

## 4. 通用传输契约

### 4.1 输入数据契约

Rust FFI 通过 [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `struct_array_to_input` 读取输入，当前实现有两个重要事实：

1. 按列位置读取，而不是按列名读取
2. 第 0 列和第 1 列都必须能下转成 `Float64Array`

因此当前 Go -> Rust 输入 record batch 的最小契约为：

| 列序号 | 语义 | Rust 读取类型 | 必填 |
|---|---|---|---|
| 0 | timestamp | `Float64Array` | 是 |
| 1 | value | `Float64Array` | 是 |

补充说明：

- Go 侧目前通过 `tableToRecord` 仅取前两列传给 Rust。
- Rust 侧不会校验列名是否为 `time` / `value`，只按位置解析。
- 如果列类型不是 `Float64Array`，当前实现会在 Rust 中 `unwrap()` 失败并触发 panic 风险。

### 4.2 history 数据契约

- `baseline_fit_predict`
- `dynamics_fit_predict`
- `rsod_forecaster`

这 3 个入口要求同时提供 `data_*` 与 `history_*` 两组 Arrow 指针。

- Go 侧包装函数会先校验 `historyFrame != nil`
- Rust 侧 `import_ffi_struct_array` 若收到空指针会直接返回 `None`，进而让 FFI 返回 `false`

`outlier_fit_predict` 允许传入空 history 指针，当前 Go 实现也确实传入 `nil`。

### 4.3 options_json 契约

- `options_json` 必须是 UTF-8 JSON 文本
- Rust 侧通过 `parse_json_options<T>` 反序列化为具体 options struct
- JSON 无法解析、字段类型不匹配或空指针时，FFI 返回 `false`

### 4.4 返回值契约

- FFI 层只返回 `bool`
- `true` 表示 Rust 已成功把结果导出到 `result_schema` 与 `result_array`
- `false` 表示导入 Arrow、解析 JSON、执行算法或导出结果的某一步失败

当前 FFI ABI 不直接返回错误消息；详细错误大多停留在 Go 包装层或 Rust 内部日志中。

## 5. 内存所有权与生命周期

### 5.1 Arrow 指针

当前实现遵循 Apache Arrow C Data Interface 的典型所有权模型：

- Go 是输入 Arrow 结构的生产者
- Rust 通过 `from_ffi` 导入输入 Arrow 数据
- Rust 是输出 Arrow 结构的生产者
- Go 通过 `ImportCRecordBatch` 导入输出结果

Go 侧当前做法：

- 输入通过 `cdata.ExportArrowRecordBatch` 导出
- 输入与输出都在 Go 侧 `defer cdata.ReleaseCArrowArray/Schema(...)`

Rust 侧当前做法：

- `import_ffi_struct_array` 从原始指针构造 `FFI_ArrowArray` 和 `FFI_ArrowSchema`
- `export_ffi_result` 通过 `to_ffi` 把 `StructArray` 导出到输出指针

### 5.2 C 字符串

`options_json` 和 `pg_dsn` 都通过 `C.CString(...)` 从 Go 传入 Rust。

当前实现中：

- `BaselineFitPredict` 会释放 `cOptsJson`
- `DynamicsFitPredict` 会释放 `cOptsJson`
- `RSODStorageInit` 会释放 `cPgDSN`
- `OutlierFitPredict` 当前直接传 `C.CString(string(optsJson))`，没有显式释放
- `RSODForecaster` 当前直接传 `C.CString(string(optsJson))`，没有显式释放

因此从接口约束上看，Go 侧应负责释放自己创建的 C 字符串；当前 `outlier` 和 `forecaster` 包装实现存在释放不一致，需要后续代码层面对齐。

## 6. 各入口函数规范

## 6.1 outlier_fit_predict

### Go 包装函数

- [pkg/rsod/rsod.go](pkg/rsod/rsod.go) 中的 `OutlierFitPredict`

### Rust 目标函数

- [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `outlier_fit_predict`
- 内部委派到 `rsod_outlier::outlier`

### 输入

- `data_schema` / `data_array`: 必填，2 列 `Float64Array`
- `history_schema` / `history_array`: 当前忽略，可以为空
- `options_json`: `OutlierOptions`

### JSON 参数

Go 结构体：

```json
{
  "model_name": "string",
  "periods": [1, 24, 168],
  "uuid": "string",
  "n_trees": 100,
  "sample_size": 256,
  "max_tree_depth": 8,
  "extension_level": 0
}
```

字段映射：

| JSON 字段 | Go 类型 | Rust 类型 |
|---|---|---|
| `model_name` | `string` | `String` |
| `periods` | `[]uint` | `Vec<usize>` |
| `uuid` | `string` | `String` |
| `n_trees` | `*int` | `Option<usize>` |
| `sample_size` | `*int` | `Option<usize>` |
| `max_tree_depth` | `*int` | `Option<usize>` |
| `extension_level` | `*int` | `Option<usize>` |

### 输出

`outlier_fit_predict` 与其他算法不同，当前返回 2 列结构：

| 列名 | 类型 | 含义 |
|---|---|---|
| `time` | `Float64` | 结果时间戳，当前实现直接把 `DetectionResult.timestamps` 转成 `f64` |
| `value` | `Float64` | 异常结果列，当前导出的是 `det.anomalies` |

Go 侧 `OutlierFitPredict` 只读取第 2 列并返回 `[]float64`，不会把 Rust 输出再组装成 `data.Frame`。

## 6.2 baseline_fit_predict

### Go 包装函数

- [pkg/rsod/rsod.go](pkg/rsod/rsod.go) 中的 `BaselineFitPredict`

### Rust 目标函数

- [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `baseline_fit_predict`
- 内部通过 `run_detector_with_history` 委派到 `rsod_baseline::baseline_detect`

### 输入

- `data_schema` / `data_array`: 必填
- `history_schema` / `history_array`: 必填
- `options_json`: `BaselineOptions`

### JSON 参数

```json
{
  "trend_type": "Daily|Weekly|Monthly|None",
  "interval_mins": 60,
  "confidence_level": 95.0,
  "allow_negative_bounds": false,
  "std_dev_multiplier": 2.0,
  "uuid": "string"
}
```

说明：

- Go 侧 `BaselineOptions` 中大部分字段不是指针，JSON 序列化时通常会直接带出具体值
- Rust 侧对应字段多为 `Option<u32>` / `Option<f64>` / `Option<bool>`，允许缺省

### 输出

当前返回 5 列结构：

| 列名 | 类型 | 可空 | 含义 |
|---|---|---|---|
| `time` | `Int64` | 否 | 当前实现导出 `DetectionResult.timestamps` |
| `baseline` | `Float64` | 是 | 基线值 |
| `lower_bound` | `Float64` | 是 | 下界 |
| `upper_bound` | `Float64` | 是 | 上界 |
| `anomaly` | `Float64` | 是 | 异常点时为原始值，正常时常为 `null` |

Rust FFI 在导出前会把 `NaN` 转为 `null`，因此后 4 列在 Arrow 中是 nullable 列。

## 6.3 dynamics_fit_predict

### Go 包装函数

- [pkg/rsod/rsod.go](pkg/rsod/rsod.go) 中的 `DynamicsFitPredict`

### Rust 目标函数

- [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `dynamics_fit_predict`
- 内部通过 `run_detector_with_history` 委派到 `rsod_baseline::dynamics::dynamics_detect`

### 输入

- `data_schema` / `data_array`: 必填
- `history_schema` / `history_array`: 必填
- `options_json`: `BaselineConfig`

### JSON 参数

```json
{
  "trend": "daily|weekly|monthly|none",
  "period_days": 90,
  "std_dev_multiplier": 2.0
}
```

字段映射：

| JSON 字段 | Go 类型 | Rust 类型 |
|---|---|---|
| `trend` | `string` | `Trend` |
| `period_days` | `int` | `Option<u32>` |
| `std_dev_multiplier` | `float64` | `f64` |

### 输出

输出 schema 与 `baseline_fit_predict` 相同：

| 列名 | 类型 | 可空 |
|---|---|---|
| `time` | `Int64` | 否 |
| `baseline` | `Float64` | 是 |
| `lower_bound` | `Float64` | 是 |
| `upper_bound` | `Float64` | 是 |
| `anomaly` | `Float64` | 是 |

当 history 为空时，Rust 会返回 cold-start 结果：

- `time` 保留
- `baseline` / `lower_bound` / `upper_bound` / `anomaly` 为 `NaN`
- FFI 导出时这些 `NaN` 会变成 `null`

## 6.4 rsod_forecaster

### Go 包装函数

- [pkg/rsod/rsod.go](pkg/rsod/rsod.go) 中的 `RSODForecaster`

### Rust 目标函数

- [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `rsod_forecaster`
- 内部通过 `run_detector_with_history` 委派到 `rsod_forecaster::forecast`

### 输入

- `data_schema` / `data_array`: 必填
- `history_schema` / `history_array`: 必填
- `options_json`: `ForecasterOptions`

### JSON 参数

```json
{
  "model_name": "string",
  "periods": [24, 168],
  "uuid": "string",
  "budget": 1.0,
  "num_threads": 1,
  "n_lags": 24,
  "std_dev_multiplier": 2.0,
  "allow_negative_bounds": false,
  "max_bin": 255,
  "iteration_limit": 200,
  "timeout": 10.0,
  "stopping_rounds": 20,
  "seed": 0,
  "log_iterations": 0
}
```

字段映射：

| JSON 字段 | Go 类型 | Rust 类型 |
|---|---|---|
| `model_name` | `string` | `String` |
| `periods` | `[]uint` | `Vec<usize>` |
| `uuid` | `string` | `String` |
| `budget` | `float32` | `Option<f32>` |
| `num_threads` | `int` | `Option<usize>` |
| `n_lags` | `int` | `Option<usize>` |
| `std_dev_multiplier` | `float64` | `Option<f64>` |
| `allow_negative_bounds` | `bool` | `Option<bool>` |
| `max_bin` | `uint16` | `Option<u16>` |
| `iteration_limit` | `*int` | `Option<usize>` |
| `timeout` | `*float32` | `Option<f32>` |
| `stopping_rounds` | `*int` | `Option<usize>` |
| `seed` | `*uint64` | `Option<u64>` |
| `log_iterations` | `*int` | `Option<usize>` |

### 输出

当前返回 5 列结构：

| 列名 | 类型 | 可空 | 含义 |
|---|---|---|---|
| `time` | `Float64` | 否 | 预测时间戳 |
| `pred` | `Float64` | 是 | 预测值 |
| `lower_bound` | `Float64` | 是 | 下界 |
| `upper_bound` | `Float64` | 是 | 上界 |
| `anomaly` | `Float64` | 是 | 异常点为原始值，正常时通常为 `null` |

与 baseline/dynamics 一样，Rust FFI 在导出前会把 `NaN` 转为 `null`。

## 6.5 rsod_storage_init

### Go 包装函数

- [pkg/rsod/rsod.go](pkg/rsod/rsod.go) 中的 `RSODStorageInit`

### Rust 目标函数

- [rsod/crates/rsod-ffi/src/lib.rs](rsod/crates/rsod-ffi/src/lib.rs) 中的 `rsod_storage_init`

### 参数

| 参数 | 类型 | 含义 |
|---|---|---|
| `trial_mode` | `bool` | `true` 表示内存 SQLite，`false` 表示 PostgreSQL |
| `pg_dsn` | `const char*` | PostgreSQL DSN，`trial_mode=true` 时可为空 |

### 返回

- `true`: 初始化成功
- `false`: 初始化失败或 Rust 内部发生 panic

当前 Rust 实现会用 `catch_unwind` 包住初始化流程，避免插件进程因存储初始化 panic 直接崩溃。

## 7. 错误语义

Go-Rust 边界当前采用“窄接口 + bool 结果”模型：

- Rust FFI 不把结构化错误返回给 Go
- FFI 失败统一表现为 `false`
- Go 包装层再把 `false` 转为固定错误文本，例如：
  - `outlier fit predict failed`
  - `baseline fit predict failed (duration: ...)`
  - `dynamics fit predict failed (duration: ...)`
  - `forecaster failed (duration: ...)`

这意味着：

- FFI 调用方可以判断成功或失败
- 但无法仅凭返回值区分是 Arrow 导入失败、JSON 解析失败、算法执行失败还是 Arrow 导出失败

## 8. 当前实现下的关键约束与已知不一致

### 8.1 输入按位置，不按列名

Rust 当前只读取第 0 列和第 1 列，因此列顺序是硬契约。

### 8.2 时间戳语义尚未完全统一

从当前代码可直接观察到：

- `baseline_detect` / `dynamics_detect` 会把输入时间戳乘以 `1000` 生成 `ts_ms`，最终经 FFI 作为 `Int64` 导出
- `rsod_forecaster` 在构造 `DetectionResult` 时直接把输入 `timestamps[i]` 转成 `i64`
- `outlier_fit_predict` 导出时也只是把 `DetectionResult.timestamps` 转成 `Float64`

因此当前代码路径里的“时间戳单位”并非所有算法完全一致。文档应以代码实际行为为准，任何后续统一都需要同步更新本文件。

### 8.3 nullable 语义仅适用于 5 列输出

`baseline` / `dynamics` / `forecast` 三类输出在 Rust FFI 中会执行 `nan_to_option`：

- `NaN` -> `null`
- 非 `NaN` -> 普通 `Float64`

而 `outlier_fit_predict` 当前 2 列输出不会做这一步转换。

### 8.4 Go 侧 CString 释放尚未完全对齐

当前 `OutlierFitPredict` 和 `RSODForecaster` 没有释放 `C.CString(...)` 创建的字符串，这属于实现层问题，不改变 ABI，但应在后续修正。

## 9. 文档更新触发条件

以下任何变更都必须同步更新本文件：

- FFI 函数签名变化
- 参数顺序变化
- 输入列顺序、类型或可空性变化
- 输出 schema 列名、列数、类型或可空性变化
- options JSON 字段变化
- 时间戳单位变化
- Arrow 所有权或释放路径变化