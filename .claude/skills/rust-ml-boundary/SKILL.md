---
name: rust-ml-boundary
description: 将机器学习流程和算法逻辑严格限制在 Rust crates 内实现。当新增或修改异常检测、预测、特征提取、模型训练或推理行为时使用。
---

# Rust 机器学习职责边界

本技能用于强制执行架构边界：机器学习逻辑只允许存在于 Rust 中，而 Go 只负责插件集成和 FFI 传输编排。

## 顶层约束关系

本文件不是 Rust 顶层规范，而是 `rust-core` 的职责边界补充。

- 只要命中本技能，必须先遵守 [/.claude/skills/rust-core/SKILL.md](../rust-core/SKILL.md) 里的全部通用 Rust 约束。
- 本文件只负责补充 **算法流程归属**、**Go 与 Rust 的职责边界**、**前端不得承载算法逻辑** 等专项约束。
- 如果本文件与 `rust-core` 出现表述重叠，以 `rust-core` 作为顶层基线，本文件只在机器学习流程归属场景中追加限制，不能放宽 `rust-core` 的要求。

## 何时使用

- 新增算法
- 修改训练、推理、评分、基线计算或特征工程
- 重构 pkg/plugin、pkg/rsod、rsod-ffi 和算法 crate 之间的代码边界
- 判断某段逻辑到底应该放在 Go 还是 Rust 中

## 必查文件

- pkg/plugin/datasource.go
- pkg/plugin/types.go
- pkg/rsod/rsod.go
- rsod/crates/rsod-ffi/src/lib.rs
- rsod/crates/rsod-*/src/

## 边界规则

1. 所有机器学习算法流程都必须实现在 rsod/crates/ 下的 Rust crates 中。
2. Go 是集成层，不是算法层。Go 可以解析查询、派生 ID、拆分时间范围、封送参数、调用 FFI、渲染结果，但不能实现模型逻辑。
3. rsod-ffi 是传输层，不是算法层。它只负责在边界上转换 Arrow 和 JSON，然后立即委派给 Rust 算法 crate。
4. 前端代码只能配置算法，不能重写算法。React 组件只负责参数与交互体验。

## Go 允许做的事情

- 解析 datasource 查询 JSON
- 填充请求默认值并派生 UUID
- 拆分 current 和 history 时间窗口
- 将 Grafana frames 转换为 Arrow 兼容结构
- 调用粗粒度 FFI 函数
- 导入 Arrow 输出并渲染 Grafana 结果帧

## Go 绝对不能做的事情

- 实现异常评分或预测数学逻辑
- 包含按算法定制的训练循环
- 执行会改变模型语义的特征工程
- 存放本应属于算法实现内部的模型特定判断分支
- 以方便调试或调用为由复制 Rust 算法代码

## Rust 必须拥有的职责

- 领域算法主流程
- 会影响模型行为的特征提取与预处理语义
- 模型训练、推理、评分、置信区间生成和异常判定逻辑
- 超出传输层健全性检查之外的算法专属校验

## 变更工作流

1. 先判断改动属于传输/集成问题，还是算法/领域逻辑问题。
2. 如果改动影响模型语义，必须优先在 Rust crate 中实现。
3. Go 侧变更应限制在新增参数、FFI 接线和结果渲染上。
4. 如果边界契约发生变化，必须同步更新接口文档。

## 最低验证要求

- go test ./pkg/...
- cd rsod && cargo test
- 如果查询结构发生变化，重新检查 docs/interfaces/ts-go-interface.md 和 docs/interfaces/go-rust-interface.md

## 反模式

- 因为不想改 Rust，就把一小段机器学习逻辑塞进 Go
- 把 rsod-ffi 扩张成第二个算法层
- 前端默认值悄悄改变模型语义，却没有对应的 Rust 变更