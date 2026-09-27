# RuVector TimesFM 接入评估

> 评估日期：2026-09-06。本文只使用项目维护者发布的仓库源码、清单、测试和官方模型说明；未实际下载权重或运行推理基准。

## 结论

**可以做为 Alert4ML 的可选、实验性长/中期预测器接入，但不建议现在替换现有的 `rsod-forecaster`，更不应作为告警的唯一判定器。**

它与本项目的 Rust 后端边界相容：当前 Grafana 插件已经通过 `rsod-backend` 直接调用 Rust 算法 crate，`ruvector-timesfm` 也提供纯 Rust API、预测中位数和 p10/p90 区间，足以转换为现有 `Pred` / `lower_bound` / `upper_bound` 结果帧。可是启用实际推理需要 Candle、模型权重和目标平台原生编译；本项目还明确记录 Grafana 插件一般不能访问文件系统。权重不随 crate 发布，真实模型测试会在本地约 814 MB safetensors 缺失时直接跳过。这让“在插件进程内 mmap 权重”成为当前最大的交付障碍。

## 本项目中的适配位置

| 项目层 | 当前事实 | TimesFM 的适配方式 |
| --- | --- | --- |
| 前端 | `forecast` 当前标注为 PerpetualBooster，并已有 history window 与固定检测间隔设置 | 增加一个显式的 `forecast_engine = perpetual | timesfm`，不要暗中改变既有 `forecast` 语义。 |
| 后端 | `rsod-backend` 负责查询、切分 Arrow frame 和渲染；算法 crate 不认识 Grafana | 在 `rsod-forecaster` 后增加一个 TimesFM adapter：`TimeSeriesInput.values` → `Vec<f32>`，结果映射回既有预测和区间字段。 |
| 发布 | Makefile 构建 x86_64/aarch64 Linux musl 静态插件 | 在这两个 target 上分别验证 Candle CPU；CUDA/Metal 是 feature-gated，不能把 GPU 支持当成现有 Grafana 插件发布物的默认能力。 |

本项目的上述边界、现有预测结果字段和 Linux musl 目标分别见 [接口说明](interfaces/README.md)、[前端类型](../src/types.ts) 与 [构建定义](../Makefile)。

## 能力与 API

该 crate 是对同仓库 `timesfm`（TimesFM 1.0 200M 的 Candle Rust port）的封装；`candle` 默认关闭，未开启时只有可序列化数据类型和纯逻辑可用。[`Cargo.toml`](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/Cargo.toml)、[`lib.rs`](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/src/lib.rs)

- `Forecaster::load(weights, device)` 一次加载转换后的 safetensors，随后 `forecast(series, horizon)` 或 `forecast_with_freq(..., freq_id)` 返回均值和九个分位数（p10…p90）。`forecast_batch` 只接受**等长**序列；不等长 Grafana series 需先对齐/重采样或逐条推理。[`forecaster.rs`](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/src/forecaster.rs)
- `detect_anomalies(history, observed, freq_id)` 先预测 `observed.len()` 步，再以 p10/p90 之外即异常；这是预测区间阈值，不是训练出的通用异常分类器。仅按两数组的较短长度评分，因而 adapter 应自行拒绝或记录长度不匹配。[`anomaly.rs`](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/src/anomaly.rs)
- 另有优化早停、向量索引 recall 重建建议和 JSON stdin/stdout CLI；这些均不对应 Alert4ML 当前的面板预测/监控告警主路径，不应纳入首个集成范围。[README](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/README.md)

## 模型、运行时与性能约束

1. **模型资产**：需要 Google `timesfm-1.0-200m` checkpoint，转换为 Candle safetensors；权重未捆绑在 crate 内。[包装 crate README](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/README.md) [基础 crate README](https://github.com/ruvnet/RuVector/blob/main/crates/timesfm/README.md)
2. **设备与依赖**：CPU 推理需 `candle`；CUDA/Metal 分别需 `cuda`/`metal` features，基础 crate 依赖 `candle-core`、`candle-nn`、`candle-transformers` 0.9。模型通过 `VarBuilder::from_mmaped_safetensors` 从路径加载，所以并非无文件/纯内存接口。[基础 crate manifest](https://github.com/ruvnet/RuVector/blob/main/crates/timesfm/Cargo.toml) [加载实现](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/src/forecaster.rs)
3. **资源与延迟**：维护者 README 声称 f32 CPU 单次预测约 45 ms、CUDA 约 4 ms；Q8 约 212 MB、Q4 约 112 MB，而真实模型测试把原 safetensors 标为 814 MB。这些是上游自述或测试常量，**不是在 Alert4ML 的 musl/Grafana 容器上复现的 SLA**；量化在 CPU 上还被源码明确说明可能更慢。[README](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/README.md) [测试](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/tests/integration.rs)
4. **并发与服务质量**：`Forecaster` 没有在公开 API 中表明 `Sync` 或并发调度策略。应把已加载模型置于受控共享状态，限制面板多 series 批量与告警并发，并为超时/内存压力保留现有预测器降级路径。

## 成熟度与正确性判断

风险等级为 **高（实验性依赖）**。这是基于源码的事实推断，而非对模型本身的否定：

- 两个 manifest 都主动称其为 “research-tier”，并放宽大量 lint；因此不宜把它当作稳定、独立版本化的生产依赖。[包装 crate manifest](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/Cargo.toml) [基础 crate manifest](https://github.com/ruvnet/RuVector/blob/main/crates/timesfm/Cargo.toml)
- 基础 TimesFM README 明列：权重与参考 PyTorch 的 parity 尚未验证、padding 相关缺陷和全 padding leading patch 的 NaN 限制。[已知限制](https://github.com/ruvnet/RuVector/blob/main/crates/timesfm/README.md#known-limitations)
- wrapper 的真实模型测试在权重缺失时打印 `SKIP` 后返回成功；其余测试主要覆盖分位数访问、区间越界标记和 rebuild 纯逻辑，不能替代 Alert4ML 的真实 metric 回测。[集成测试](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/tests/integration.rs)

## 许可

仓库根 `LICENSE` 和 workspace `license` 都是 **MIT**，且该 crate 使用 `license.workspace = true`；然而两个 crate README 的 badge/文字写 **Apache-2.0**。这是直接可见的冲突，接入前必须以计划锁定的 crates.io 发布包、commit 和模型权重各自的许可做法律审查，不能只依据 README badge。模型权重也不是该 Rust 仓库的资产，须另行遵守其官方模型页所列条款。[根许可证](https://github.com/ruvnet/RuVector/blob/main/LICENSE) [workspace 清单](https://github.com/ruvnet/RuVector/blob/main/Cargo.toml) [crate README 的 Apache 声明](https://github.com/ruvnet/RuVector/blob/main/crates/ruvector-timesfm/README.md)

## 推荐的决策与最小验证门槛

建议采用“**可选 POC，保留现有算法**”的决策。先在独立 feature（例如 `timesfm`）下以锁定 git commit 或已核验的 crate 版本接入，不要引入整个 RuVector workspace/path dependency。部署时由管理员提供只读模型卷，或改造成明确受支持的受控模型服务；若 Grafana 环境仍禁止该路径，则停止进程内方案。

在进入生产前，至少完成：

1. 在 x86_64 与 aarch64 的 `*-unknown-linux-musl` 发布构建实际启用 Candle，并验证启动、内存、单/批 series 延迟和并发上限。
2. 以 Prometheus 等真实历史数据进行 rolling-origin 回测：与现有 Perpetual、naive seasonal baseline 比较预测误差、p10/p90 覆盖率、异常 precision/recall 和告警噪声。
3. 明确时间序列重采样、缺失值/NaN、变长序列和 `freq_id` 的映射规则；测试历史不足、上游数据断档、权重缺失与模型加载失败时的安全降级。
4. 固定依赖与权重校验和，完成上节许可冲突和第三方模型许可审查；禁止运行时任意下载权重。

若以上门槛通过，它适合补充“零样本、可给预测区间”的预测能力；若目标是快速、稳定的实时告警，当前本项目已有的轻量预测/基线算法仍是更低风险的默认实现。
