# Alert4ML 仓库级入口

本文件是仓库级总入口，用于帮助 agent 在开始工作时快速判断：

1. 这是一个什么项目
2. 代码主要分布在哪些层
3. 当前任务应该优先命中哪个 skill

Plugin ID: `ibumblebee-alert4ml-datasource`

## 技术栈

- 前端：TypeScript + React 18（Grafana UI components）
- 后端：Go 1.25 + CGO -> Rust FFI
- ML 引擎：Rust workspace（`rsod/crates/`）
- 构建：Webpack、Mage、Cargo（使用 zigbuild 交叉编译）

## 项目结构

```text
pkg/              Go backend（plugin entry、query handler、data conversion）
rsod/crates/      Rust ML engine（outlier、forecaster、baseline、classifier、storage）
src/              TypeScript frontend（components、datasource、types）
src/plugin.json   Plugin metadata and capabilities
tests/            Playwright e2e tests
.config/          Grafana scaffolded configs（webpack、jest、eslint、prettier）
.github/          CI/CD workflows and custom build actions
provisioning/     Grafana provisioning configs
```

## 构建命令

### 全量构建

```bash
make all
make all-platforms
```

### 单独构建

```bash
# Frontend
npm install
npm run build
npm run dev

# Go backend
mage build:linux
mage build:linuxARM64

# Rust ML engine
mage build:rsAMD64
mage build:rsARM64

# Combined
mage build:all
mage build:allPlatforms
```

### 清理

```bash
make clean
mage clean
```

## 测试与检查

```bash
npm run test:ci
go test ./pkg/...
cd rsod && cargo test
npm run e2e
npm run typecheck
npm run lint
npm run lint:fix
```

## 本地开发

```bash
make -f Makefile.cross.local
```

Grafana runs with unsigned plugin loading enabled. Delve debugger available on port 2345.

Plugin logs: `GF_LOG_FILTERS=plugin.ibumblebee-alert4ml-datasource:debug`

## 项目概览

- 这是一个 Grafana datasource 插件项目，不是通用 Web 应用。
- 前端位于 `src/`，负责查询编辑器、配置编辑器和 Grafana 插件 UI。
- Go 后端位于 `pkg/`，负责 Grafana 插件协议、查询编排、FFI 调用和结果渲染。
- Rust 代码位于 `rsod/crates/`，负责机器学习算法、FFI 边界、存储与共享核心类型。
- Rust 产物面向 `linux/amd64` 与 `linux/arm64` 的 musl 静态构建，统一使用 `cargo zigbuild`。

## 架构说明

- Go backend 接收 Grafana queries，通过 FFI 调用 Rust ML engine（`pkg/rsod/`）。
- Rust workspace 当前包含 `rsod-outlier`、`rsod-forecaster`、`rsod-baseline`、`rsod-classifier` 等专用 crate。
- 前端组件与能力映射：Baseline、Forecast、Dynamics、Outlier detection。
- 所有二进制默认静态链接（musl）以确保跨平台兼容。
- Cross-compilation 使用 zig 作为 C compiler。

## CI/CD

- Release workflow 由版本 tag（`v*`）触发。
- Pipeline：frontend build -> backend build（amd64/arm64 matrix）-> GitHub release artifacts。

## 通用约定

- Go：standard library style，`pkg/` layout
- Rust：workspace with `rsod-` prefix for crates
- TypeScript：Grafana plugin conventions，`@grafana/eslint-config`
- Node 22，npm 10.9，Rust nightly-2025-12-05，Zig 0.15.2

## 工作入口判断

在开始修改前，先判断任务主要落在哪一层：

1. 如果任务涉及 Grafana 插件 UI、查询编辑器、plugin.json、健康检查、日志、配置处理或兼容性，优先使用 [skills/grafana-plugin-practices/SKILL.md](skills/grafana-plugin-practices/SKILL.md)。
2. 如果任务涉及 `rsod/` 目录下的 Rust 代码，先使用 [skills/rust-core/SKILL.md](skills/rust-core/SKILL.md) 作为顶层基线。
3. 如果 Rust 任务还涉及专项场景，再在 `rust-core` 基础上追加对应专项 skill。

## Rust Skills 索引

所有 Rust 相关任务都必须先遵守 [skills/rust-core/SKILL.md](skills/rust-core/SKILL.md)。

### 顶层基线

- [skills/rust-core/SKILL.md](skills/rust-core/SKILL.md)
  - 适用范围：所有 `rsod/` 目录修改
  - 负责内容：Rust workspace 结构、模块拆分、跨架构构建、`cargo zigbuild`、错误处理、`Result` 约束、依赖引入审查

### 专项补充

- [skills/rust-algorithm-crate-layout/SKILL.md](skills/rust-algorithm-crate-layout/SKILL.md)
  - 适用范围：新增算法、拆分 crate、控制 `lib.rs` 膨胀、调整模块布局
  - 负责内容：新算法 crate 的目录结构、模块职责、依赖方向

- [skills/rust-ml-boundary/SKILL.md](skills/rust-ml-boundary/SKILL.md)
  - 适用范围：训练、推理、特征提取、评分、预测、异常检测相关改动
  - 负责内容：机器学习流程只在 Rust 中实现，Go 和前端不得承载算法逻辑

- [skills/arrow-safe-ffi/SKILL.md](skills/arrow-safe-ffi/SKILL.md)
  - 适用范围：Go-Rust FFI、Arrow Schema、时间戳单位、内存释放、跨语言数据交换
  - 负责内容：Arrow C Data Interface、FFI 安全、Schema 契约、边界内存所有权

## Grafana Plugin Skill 索引

- [skills/grafana-plugin-practices/SKILL.md](skills/grafana-plugin-practices/SKILL.md)
  - 适用范围：`src/`、`pkg/plugin/`、`pkg/sdk/`、`src/plugin.json`、测试和兼容性改动
  - 负责内容：Grafana datasource 插件最佳实践、前端 UI 约束、plugin metadata、运行时兼容性、日志与配置安全

## 修改前最小检查

- 先确认当前任务主要落在前端、Go 后端、Rust 算法、还是 Go-Rust 边界。
- 如果是 Rust 任务，先读 `rust-core`，再决定是否联读其他专项 skill。
- 如果是跨层任务，按“插件层 → Go 编排层 → Rust 边界层 → Rust 算法层”的顺序梳理影响范围。
- 如果任务影响公共契约，联动检查 `docs/interfaces/ts-go-interface.md` 和 `docs/interfaces/go-rust-interface.md`。

## 约束原则

- 不要把机器学习逻辑移入 Go 或前端。
- 不要绕过 Arrow C Data Interface 自创 Go-Rust 交换协议。
- 不要把 Rust 专项规则分散成并列顶层规范；Rust 顶层基线统一由 `rust-core` 提供。
- 不要把本文件写成重复的详细规范；详细规则应下沉到对应 skill 中维护。
- 不允许 agent 自行向 GitHub 执行 `git push` 或任何等效远端发布操作；涉及远端提交、推送、发版前，必须先完成严格审核并获得用户的明确同意。
