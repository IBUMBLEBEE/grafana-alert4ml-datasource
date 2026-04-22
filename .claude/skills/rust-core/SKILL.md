---
name: rust-core
description: 当涉及 rsod/ 目录修改时使用。涵盖核心算法实现、数据结构、公共接口、错误处理、配置定义等。任何直接影响模型行为的 Rust 代码都必须遵守此规范。
---

# Rust 核心开发规范（架构与风格）

## 0. 关联 Skills 与联读要求

### 0.1 算法实现来源与授权约束

- 当前仓库中的机器学习算法实现必须优先落在 Rust 端，不能把会影响模型语义的逻辑转移到 Go、FFI 适配层或前端。
- 当需要参考现有 Rust 生态中的算法设计、训练流程、特征处理或预测实现时，优先参考以下来源：
  - `anofox-forecast`: https://github.com/sipemu/anofox-forecast
  - `Perpetual ML`: https://perpetual-ml.com/
  - `linfa`: https://github.com/rust-ml/linfa
  - `augurs`: https://github.com/grafana/augurs
- 这些来源只作为算法思路、模块组织和依赖选型的参考，不能因此绕过本仓库既有的 workspace 分层、musl 交叉编译、错误处理和 FFI 边界约束。
- 如果实现某个需求需要新增承载算法语义的 Rust 模块、新增算法 crate、扩展 workspace 成员，或引入新的算法型第三方依赖，必须先获得用户授权，再继续实施。

本文件定义 rsod/ 目录下 Rust 核心代码的通用架构与风格约束，但以下场景不得只依赖本文件，必须同时读取对应的其他 SKILL.md：

- 当涉及 **新增算法**、新增 crate、拆分过大的 `lib.rs`、调整算法模块布局时，必须同时读取 [.claude/skills/rust-algorithm-crate-layout/SKILL.md](../rust-algorithm-crate-layout/SKILL.md)。
- 当涉及 **Go ↔ Rust FFI 边界**、Arrow Schema、ArrowArray、ArrowSchema、时间戳单位、零拷贝传输、内存释放、`extern "C"` 接口签名时，必须同时读取 [.claude/skills/arrow-safe-ffi/SKILL.md](../arrow-safe-ffi/SKILL.md)。
- 当涉及 **模型行为**、训练流程、推理流程、异常检测、预测、特征提取、评分逻辑归属时，必须同时读取 [.claude/skills/rust-ml-boundary/SKILL.md](../rust-ml-boundary/SKILL.md)。

优先级原则：

1. 所有 Rust 相关 skills 统一以 `rust-core` 作为顶层基线；凡是命中 Rust 专项 skill，必须先遵守 `rust-core`。
2. `rust-core` 负责 Rust 代码的通用组织、工程约束、风格基线与跨架构构建要求。
3. `rust-algorithm-crate-layout` 只负责新增算法时的 crate 与模块布局约束。
4. `arrow-safe-ffi` 只负责 Go-Rust 跨语言边界与 Arrow C Data Interface 的安全约束。
5. `rust-ml-boundary` 只负责算法职责边界，防止算法流程泄漏到 Go 或 FFI 层。

如果多个文件同时适用，必须联合遵守，不能以 `rust-core` 为由覆盖更具体的边界规则。

## 1. 项目组织与模块化
**AI 必须严格遵守此目录结构，禁止将所有代码堆积在 `lib.rs` 中。**

### 1.1 分层架构
当前仓库不是单 crate 目录布局，而是 **Rust workspace + 多 crate 分层**。编写或修改 Rust 代码时，必须优先遵守现有 `rsod/crates/` 结构，而不是臆造一个新的目录体系。

- **`rsod-core/` (核心抽象层)**：
  - 存放跨算法共享的公共数据结构、结果类型、常量、基础 trait 与通用接口。
  - **原则**：只定义领域抽象和共享契约，不掺入具体算法流程。
- **`rsod-outlier/`、`rsod-forecaster/`、`rsod-baseline/`、`rsod-classifier/` (算法实现层)**：
  - 存放具体算法实现、训练流程、推理流程、评分逻辑与算法相关配置。
  - **原则**：算法行为在对应 crate 内聚，不向 `rsod-ffi` 或 Go 层泄漏实现细节。
- **`rsod-ffi/` (边界层)**：
  - 专门处理 `#[no_mangle]`、`extern "C"`、Arrow FFI 转换、JSON options 解码与跨语言边界适配。
  - **原则**：这一层只做边界转换与委派，严禁承载业务算法流程。
- **`rsod-storage/`、`rsod-utils/` (基础设施层)**：
  - 分别承载存储能力与通用工具逻辑。
  - **原则**：保持职责单一，避免反向依赖算法 crate 的私有实现。

### 1.2 模块声明
- `src/lib.rs` 必须清晰声明模块、导出稳定公共接口，并保持为薄入口文件。
- `src/lib.rs` 可以做的事情：`mod` 声明、`pub use` 重导出、少量公开入口函数拼装。
- `src/lib.rs` 不可以做的事情：塞入完整训练流程、推理流程、特征工程、错误处理细节、Arrow/FFI 传输细节。
- 当某个 crate 的 `lib.rs` 开始同时承担配置、模型、训练、推理、序列化、FFI 适配等职责时，必须立即拆分模块。

### 1.3 当前仓库的推荐模块划分

在单个算法 crate 内，优先按职责拆分，而不是把所有代码直接写进一个文件：

- `config.rs`：算法配置与默认值
- `error.rs`：crate 内部错误类型
- `input.rs`：输入数据适配与校验
- `model.rs` 或 `pipeline.rs`：核心算法对象或主流程
- `train.rs`：训练逻辑
- `predict.rs` 或 `score.rs`：推理、评分、异常判定逻辑
- `tests.rs` 或 `tests/`：面向行为的测试

如果 crate 很小，可以减少文件数量，但不能牺牲职责边界。

### 1.4 禁止事项

- 禁止把整个算法实现塞进 `src/lib.rs`
- 禁止把算法主流程写进 `rsod-ffi`
- 禁止在 `rsod-core` 中混入具体算法实现
- 禁止让工具模块反向依赖具体算法 crate
- 禁止为了“方便调用”在多个 crate 中复制同一份算法逻辑
- **禁止 `unwrap()`**：在生产代码中严禁使用 `unwrap()`，必须处理 `Option` 和 `Result`。
-  **禁止克隆大数据**：避免在循环中 `clone()` 大数据，使用引用 `&`。
- **禁止忽略架构兼容性**：新增 Rust 代码实现、引入新的第三方 crate、升级关键依赖时，必须确认其可用于 `amd64` 与 `arm64`，并与当前仓库的 `x86_64-unknown-linux-musl`、`aarch64-unknown-linux-musl` 目标兼容。
- **禁止默认使用 `cargo build` 作为发布构建路径**：本仓库的 Rust 产物面向跨架构静态构建，发布与交叉编译必须使用 `cargo-zigbuild`，不能绕过既有构建链。

### 1.5 自动化钩子 (Automation)
- **构建脚本**：参考 `Magefile.go`。在 `cargo build` 前必须执行 `cargo fmt --check`。
- **Clippy**：代码必须通过 `cargo clippy -- -D warnings`。
- **跨架构要求**：任何新增 Rust 实现或第三方依赖，在合入前都必须至少从规则层确认同时支持 `linux/amd64` 与 `linux/arm64`，对应 Rust 目标分别为 `x86_64-unknown-linux-musl` 和 `aarch64-unknown-linux-musl`。
- **强制构建方式**：Rust 项目编译必须使用 `cargo zigbuild`，与仓库现有 Makefile 和 Magefile 保持一致。涉及发布、交叉编译或构建脚本调整时，应优先参考 `Makefile` 中的 `build-rs-amd64`、`build-rs-arm64` 以及 `Magefile.go` 中的 `RsAMD64`、`RsARM64`。
- **推荐验证命令**：
  - `cargo fmt --check`
  - `cargo clippy -- -D warnings`
  - `cargo test`
  - `cargo zigbuild --release --target x86_64-unknown-linux-musl`
  - `cargo zigbuild --release --target aarch64-unknown-linux-musl`

## 2. 错误处理与 Result 约束

### 2.1 总体原则

- Rust 代码必须优先返回 `Result`，而不是用 `panic!`、`unwrap()`、`expect()` 或隐式失败来表达正常错误路径。
- 业务错误、配置错误、输入校验错误、模型状态错误、存储错误、FFI 转换错误，都必须作为显式错误向上返回，而不是在底层直接中断进程。
- 只要错误可能由外部输入、运行时状态、数据质量、依赖行为或跨语言边界触发，就必须视为可恢复错误并返回 `Result`。

### 2.2 推荐错误类型

- 跨 crate 共享的领域错误，优先收敛到 `rsod-core` 中的错误类型与 `Result` 别名。
- 单个算法 crate 内部允许定义本 crate 的 `error.rs`，但应保持可组合、可传播，并在公共边界上转换为稳定错误类型。
- 禁止把 `String` 作为长期公共错误模型到处传播；只有在非常薄的兼容层或临时基础设施接口中，才允许短期存在。
- `Box<dyn Error>` 只能作为过渡性边界类型使用，不能成为整个 crate 的默认错误设计。
- 新增公共 API 时，优先定义明确错误枚举，而不是继续扩散模糊错误类型。

### 2.3 unwrap/expect/panic 规则

- 生产代码中默认禁止 `unwrap()`。
- 生产代码中默认禁止 `expect()`，除非是进程启动期、测试专用路径之外的绝对不变量，并且错误消息具有明确诊断价值。
- 生产代码中默认禁止 `panic!()` 作为输入校验、算法状态控制或普通错误处理手段。
- 测试代码、测试辅助代码、示例代码中允许使用 `unwrap()` / `expect()`，但应保持错误意图清晰。
- 文档示例如果使用 `unwrap()`，必须确保其用途是演示 API，而不是暗示生产代码应采用该模式。

### 2.4 错误传播规则

- 能使用 `?` 的地方优先使用 `?`，不要手工层层匹配后再丢失上下文。
- 将底层错误上抛时，必须保留足够上下文，让调用方能判断是输入问题、依赖问题、存储问题还是模型状态问题。
- FFI 层不得吞掉 Rust 错误语义后仅留下模糊失败；如果当前 ABI 只能返回布尔值，至少要在 Rust 内部保持错误分类清晰，方便后续演进与测试。
- 算法 crate 不应把错误细节泄漏为 Go 或 Grafana 特有概念；边界转换应发生在更外层。

### 2.5 仓库级具体要求

- `rsod-core` 中已有错误类型和 `Result` 别名时，新代码应优先复用，而不是重新发明并行的基础错误模型。
- 新增 crate 时，应尽早建立 `error.rs` 或等价错误模块，而不是等到错误处理扩散后再回补。
- 输入校验失败、空数据、NaN/Infinity、模型未训练、模型加载失败、Schema 不匹配、时间戳单位不匹配等情况，都应返回显式错误，而不是直接 `panic!`。
- 当第三方库只能返回其自身错误类型时，应在 crate 边界完成包装或转换，避免公共 API 暴露难以维护的底层细节。

### 2.6 审查清单

- 公共函数是否返回了明确的 `Result`
- 错误类型是否稳定且可理解
- 是否错误地把 `String` 或 `Box<dyn Error>` 扩散为默认公共接口
- 是否在生产路径中残留 `unwrap()` / `expect()` / `panic!()`
- 是否保留了足够的错误上下文，便于定位输入、存储、算法、FFI 或依赖问题

## 3. 依赖引入审查规则

### 3.1 总体原则

- 新增 Rust 第三方依赖不是普通实现细节，而是架构决策，必须审查其对 **跨架构构建**、**musl 静态链接**、**FFI 边界**、**算法职责边界** 和 **编译复杂度** 的影响。
- 只有在标准库、现有 workspace 依赖或仓库内已有 crate 无法合理解决问题时，才允许新增第三方库。
- 引入依赖时，优先选择职责单一、维护活跃、API 稳定、对 `x86_64-unknown-linux-musl` 与 `aarch64-unknown-linux-musl` 兼容性明确的库。

### 3.2 工作区依赖规则

- 能收敛到 `rsod/Cargo.toml` 的 `[workspace.dependencies]` 时，优先在那里统一声明版本，而不是让多个 crate 私自漂移。
- 新增共享依赖时，应优先考虑它是否会被多个 crate 复用；如果会，优先放入 workspace 依赖统一管理。
- 禁止在多个 crate 中随意引入同一依赖的不同版本，除非有明确、可解释且短期无法消除的技术原因。
- 引入依赖后，应检查它是否会破坏现有 crate 的单一职责边界。

### 3.3 原生依赖与系统依赖审查

- 任何带有 `-sys`、原生编译脚本、C/C++/汇编代码、系统库绑定、静态链接需求的依赖，都必须额外审查其在 `cargo zigbuild` 下对 amd64 与 arm64 的可用性。
- 涉及 SQLite、压缩库、加密库、SIMD、CPU 特性检测、汇编优化路径的依赖，必须明确确认其不会破坏 musl 静态构建。
- 如果依赖需要额外系统库、动态链接、特定 glibc 行为或宿主机环境假设，则默认不适合本仓库，除非先证明其可被当前交叉编译链稳定支持。
- 优先选择已经在本仓库中被验证过的构建模式，例如 `rusqlite` 的 `bundled` 特性这类更可控的依赖方式。

### 3.4 Feature 与体积控制

- 新增依赖时，必须显式审查默认 features；能关闭的默认 features 就不要无条件开启。
- 当依赖提供 `default-features = false` 的可裁剪模式时，应优先评估最小功能集，而不是直接引入全量能力。
- 禁止仅为一个很小的辅助功能引入重量级依赖，尤其是会拉入大量传递依赖、原生构建脚本或多套压缩/序列化后端的库。

### 3.5 架构与边界约束

- 算法 crate 引入的依赖必须服务于算法实现本身，不能顺带把 Go、Grafana、FFI 或存储边界概念带进算法层。
- `rsod-ffi` 中新增依赖必须严格服务于 FFI 转换、Arrow 交换或必要的边界支撑，不能把算法依赖错误地堆到 FFI 层。
- `rsod-core` 中新增依赖必须极其克制，避免让核心抽象层背负沉重实现依赖。
- 新依赖若会改变公共数据结构、错误模型、序列化格式或 FFI 输出结构，必须视为公共契约变更处理。

### 3.6 引入前审查清单

- 这个问题是否真的需要新依赖，而不是标准库或现有 crate 就能解决
- 依赖是否兼容 `x86_64-unknown-linux-musl` 与 `aarch64-unknown-linux-musl`
- 依赖是否能在 `cargo zigbuild` 下稳定构建
- 是否包含原生代码、`build.rs`、动态链接假设或额外系统库要求
- 是否应放入 `[workspace.dependencies]` 统一管理
- 是否会把实现细节错误地下沉到 `rsod-core` 或错误地上浮到 `rsod-ffi`
- 是否默认开启了过多 features

### 3.7 最低验证要求

- 修改依赖后至少执行一次 `cargo test`
- 修改依赖后至少执行一次 `cargo zigbuild --release --target x86_64-unknown-linux-musl`
- 修改依赖后至少执行一次 `cargo zigbuild --release --target aarch64-unknown-linux-musl`
- 如果依赖影响 FFI、Schema 或公共接口，联动检查相关 skill 与接口文档