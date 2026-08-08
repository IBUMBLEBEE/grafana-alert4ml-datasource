# vendored grafana-plugin-sdk 0.5.0

crates.io 版本 `grafana-plugin-sdk = "0.5.0"` 的本地副本，带一个反序列化修复。

## 为什么 vendor

Grafana 的 frame JSON（`FrameJSON`）序列化 `schema.name` / `schema.refId` 时带
`omitempty`：frame name 为空（prometheus、testdata 等上游的 frame name 恒为空）
时 `name` 键被整体省略。SDK 0.5.0 的 `Schema` 结构体却把这两个字段定义为必填，
导致通过 `/api/ds/query` 代理查询时反序列化失败：

```
query A failed: datasource query: failed to decode /api/ds/query response:
error decoding response body
```

（Go 后端不受影响：Go SDK 的 `json:"name,omitempty"` 反序列化空值时静默置空。）

## 本地的改动

`src/data/frame/de.rs` 的 `Schema` 结构体：

```rust
#[serde(default)]
name: String,
#[serde(default)]
ref_id: String,
```

仅此一处（2 行）。其余内容与 crates.io 0.5.0 一致。

## 何时移除

上游修复后（新的 crates.io 版本），删除本目录和 `rsod/Cargo.toml` 中的
`[patch.crates-io]` 段，恢复 `grafana-plugin-sdk = "<新版本>"`。
