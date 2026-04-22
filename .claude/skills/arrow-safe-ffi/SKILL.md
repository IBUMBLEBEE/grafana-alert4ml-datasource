---
name: arrow-safe-ffi
description: 强制使用 Apache Arrow C Data Interface 与安全的 Go-Rust FFI 边界。当修改 pkg/rsod、rsod-ffi、Arrow Schema、时间戳单位或跨语言数据交换时使用。
---

# Arrow 安全 FFI 规范

本技能定义当前仓库中 Go 与 Rust 之间唯一被允许的进程内数据交换模型。

## 顶层约束关系

本文件不是 Rust 顶层规范，而是 `rust-core` 在跨语言边界上的专项补充。

- 只要命中本技能，必须先遵守 [/.claude/skills/rust-core/SKILL.md](../rust-core/SKILL.md) 里的全部通用 Rust 约束。
- 本文件只负责补充 **Arrow C Data Interface**、**FFI 内存所有权**、**Schema 契约**、**Go-Rust 边界安全** 等专项规则。
- 如果本文件与 `rust-core` 出现表述重叠，以 `rust-core` 作为顶层基线，本文件只在 FFI 与 Arrow 边界场景中追加更具体限制，不能覆盖或降低 `rust-core` 的要求。

## 何时使用

- 修改 pkg/rsod/rsod.go
- 修改 rsod/crates/rsod-ffi/src/lib.rs
- 修改 docs/interfaces/go-rust-interface.md
- 修改 Arrow Schema、列顺序、可空性、时间戳单位或 FFI 函数签名
- 新增跨越 Go-Rust 边界的算法

## 必查文件

- pkg/rsod/rsod.go
- rsod/crates/rsod-ffi/src/lib.rs
- docs/interfaces/go-rust-interface.md
- pkg/plugin/datasource.go

## 仓库规则

1. Go 与 Rust 必须通过 Apache Arrow 和 Arrow C Data Interface 交换时序数据，禁止使用临时二进制格式、CSV、逐行结构体或自定义序列化层。
2. 必须以 ArrowSchema 与 ArrowArray 语义作为边界契约。Schema 与数组的生命周期规则是实现契约的一部分，不是可选细节。
3. 在可行情况下优先使用零拷贝交换。除非正确性或所有权约束要求，否则禁止增加不必要的 buffer 拷贝。
4. 跨边界导出的 Arrow 数据必须视为不可变。
5. 保持 FFI 表面尽可能窄。结构化参数通过 JSON C 字符串传递，列式数据通过 Arrow batch 传递，禁止为算法内部流程扩张 ABI。

## 安全规则

1. 生产者拥有 ArrowSchema 与 ArrowArray 可达的所有内存。
2. 消费者只能通过 release callback 释放导出的基结构，不能手工释放由生产者拥有的子结构或 buffers。
3. 已释放结构必须通过 NULL release callback 标识，之后不得再次访问。
4. 禁止存储自引用指针，或依赖 Arrow 结构体固定内存地址的簿记信息。
5. Go 侧创建的任何 CString 或裸指针都必须有明确的生命周期管理策略，禁止在重复查询执行中泄漏 C 分配。
6. rsod-ffi 中的 unsafe Rust 必须最小化并局限于 FFI 转换边界，算法 crate 不得吸收原始 C 边界问题。

## Schema 规则

1. 列顺序是契约的一部分。Rust 代码可能按索引读取，因此 Go 侧 frame 转换必须保持文档约定的布局。
2. 时间戳单位是契约的一部分。如果某个算法使用秒而另一个算法使用毫秒，这种差异必须在代码和文档中显式表达。
3. 可空性是契约的一部分。禁止在未同步更新两端实现与接口文档的情况下，静默改变 NaN、null、0 或缺失 buffer 的解释方式。
4. 任何输出 schema 变更，都必须在同一个改动中同步更新实现和 docs/interfaces/go-rust-interface.md。

## 允许的实现模式

1. Go 准备 data.Frame 数据并将其转换为 Arrow record batch。
2. Go 通过 arrow-go 的 cdata helper 导出 Arrow record batch。
3. Go 将 JSON options 与 Arrow 指针传入小而稳定的 FFI 入口函数。
4. rsod-ffi 将 Arrow 转换为 Rust 原生类型输入，委派给 Rust 算法 crate，然后把结果重新导出为 Arrow。
5. Go 导入 Arrow 结果并渲染 Grafana frames。

## 禁止模式

- 用手写的逐行 C 结构体替换 Arrow 作为算法载荷
- 把算法逻辑嵌进 rsod-ffi，而不是委派给算法 crate
- 因为“逻辑很小”就在 Go 侧加入算法循环
- 修改 FFI schema 却不更新 docs/interfaces/go-rust-interface.md
- 使用本地文件作为 Go 与 Rust 的交换层

## 最低验证要求

- go test ./pkg/...
- cd rsod && cargo test
- 重新检查 docs/interfaces/go-rust-interface.md 中 schema、参数和时间戳定义是否仍然准确

## 反模式

- Go 与 Rust 之间出现隐蔽的时间戳单位漂移
- 内存所有权分散在两端，却没有统一释放路径
- 在同一个文件中混杂传输关注点、Arrow 转换和算法逻辑