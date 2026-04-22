---
name: rust-algorithm-crate-layout
description: 约束新增 Rust 算法的 crate 与模块布局。当新增 rsod 算法 crate 或重构现有 Rust 算法代码时使用。
---

# Rust 算法 Crate 布局规范

本技能定义在当前 Rust workspace 中新增机器学习算法时必须遵守的目录与模块结构。

## 顶层约束关系

本文件不是 Rust 顶层规范，而是 `rust-core` 的专项补充。

- 只要命中本技能，必须先遵守 [/.claude/skills/rust-core/SKILL.md](../rust-core/SKILL.md) 里的全部通用 Rust 约束。
- 本文件只负责补充 **新增算法 crate**、**模块拆分**、**lib.rs 控制**、**crate 依赖方向** 等专项要求。
- 如果本文件与 `rust-core` 出现表述重叠，以 `rust-core` 作为顶层基线，本文件只在更具体的新增算法与 crate 布局场景中追加限制，不能降低 `rust-core` 的要求。

## 何时使用

- 在 rsod/crates/ 下新增 crate
- 拆分过大的 lib.rs
- 将算法代码从 rsod-ffi 迁移到独立 crate
- 为了可维护性重构现有算法 crate

## 新增前置授权

1. 如果要新增承载机器学习语义的模块、在 rsod/crates/ 下新增 `rsod-<algorithm>` crate、或把新的算法 crate 加入 Rust workspace，必须先获得用户明确授权。
2. 如果只是修改现有算法 crate 的内部实现，可继续在既有边界内完成，但不能借机扩张出新的算法模块层级。
3. 如果新增模块的必要性来自外部算法参考库或新的第三方依赖，同样要先说明原因并等待用户确认。

## 必查文件

- rsod/Cargo.toml
- rsod/crates/
- rsod/crates/rsod-ffi/src/lib.rs
- docs/interfaces/go-rust-interface.md

## 目录规则

1. 每个新算法都必须以独立 crate 的形式存在于 rsod/crates/ 下，并遵循 rsod-<algorithm> 命名模式。
2. 禁止将新的算法逻辑直接实现到 rsod-ffi 中。
3. 禁止把完整算法实现堆进 src/lib.rs。
4. lib.rs 必须保持为薄入口文件，只负责组织模块、定义公共 API、重导出稳定类型。

## 新算法 crate 的推荐结构

以下结构应作为默认起点。只有在 crate 的确非常简单时，才允许适度精简：

```text
rsod/crates/rsod-<algorithm>/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── config.rs
    ├── error.rs
    ├── input.rs
    ├── pipeline.rs
    ├── train.rs
    ├── predict.rs
    └── tests.rs
```

如果算法不需要显式区分 train 和 predict 阶段，也必须保持模块化，可改用 model.rs、transform.rs、scoring.rs、seasonal.rs 等面向领域职责的模块。

## lib.rs 规则

1. lib.rs 不能成为所有结构体、辅助函数和算法步骤的堆放地。
2. lib.rs 只应聚焦于模块声明、公共类型导出和少量公开入口函数。
3. 大型辅助函数、特征变换、训练流程和评分逻辑必须拆分到专用模块中。

## 依赖规则

1. 算法 crate 只能向内依赖 rsod-core、rsod-utils 等共享 crate，不能向外依赖 Grafana、Go 或 FFI 相关概念。
2. rsod-ffi 可以依赖算法 crate，但算法 crate 不能反向依赖 rsod-ffi。
3. 保持 crate 单一职责：核心类型放在 rsod-core，存储能力放在 rsod-storage，传输边界放在 rsod-ffi，算法行为放在 rsod-<algorithm>。

## 新增算法的审查清单

1. 新 crate 已在 rsod/crates/ 下创建，并使用 rsod- 前缀命名
2. 如有需要，Cargo workspace 已同步更新
3. 公共 API 从新 crate 暴露，而不是从 rsod-ffi 暴露
4. rsod-ffi 只负责 FFI 输入输出映射与委派，不承载算法主体
5. 如果引入了新的 FFI 入口或 Schema，docs/interfaces/go-rust-interface.md 已同步更新

## 禁止模式

- 在 rsod-ffi/src/lib.rs 中实现完整算法
- 让 lib.rs 膨胀成数百行的混合实现文件
- 在同一个模块里混杂配置解析、模型训练、特征提取和 Arrow 传输
- 在算法 crate 中引入 Go 或 Grafana 特有概念

## 最低验证要求

- cd rsod && cargo test
- 确认 rsod-ffi 到 rsod-<algorithm> 的边界仍然清晰
- 重新检查任何变更过的 FFI 签名或 Schema 文档