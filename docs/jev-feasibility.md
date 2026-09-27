# Jev 用于分类与模型路由的可行性评估

> 评估日期：2026-09-22。本文只把 TypeSafe 官方网站、文档、协议、官方 GitHub 仓库及其源码作为 Jev 能力依据；未申请 API key，也未运行线上推理或质量基准。

## 结论

**Jev 可以用于固定候选之间的语义分类、模型选择、路由和定性打分，但不适合替换 Alert4ML 现有的数值时间序列分类器。建议只做可选的 shadow POC，现有本地分类与路由必须保留为默认路径和故障回退。**

首先要纠正“TypeSafe 开源了 Jev”这一前提：用户给出的 [`yibie/jev-engineering-zh`](https://github.com/yibie/jev-engineering-zh#jev-%E5%B7%A5%E7%A8%8B%E5%AD%A6%E4%B8%BA-coding-agent-%E8%80%8C%E4%BD%9C) 是第三方中文翻译/独立汇编，仓库自己明确声明与 TypeSafe 无隶属关系、未获背书；它是文档，不是可安装的模型、SDK 或运行时。Jev 本身是 TypeSafe 托管的商业 API 服务。官方开源的是 MIT 许可的 Python 与 JavaScript/TypeScript 客户端 SDK，而不是模型权重、训练代码或自托管推理实现。TypeSafe 的客户协议也把产品定义为 TypeSafe-hosted Web/API 服务，并只授予有限的访问和集成许可。[官方 SDK 列表](https://docs.typesafe.ai/sdk) · [官方 GitHub 组织](https://github.com/typesafe-ai) · [客户协议 §§1–2](https://typesafe.ai/legal/mca)

对本项目更关键的是任务形状：Jev 擅长“给定文字或结构化业务状态，从预先声明的选项中判断”，而现有 `rsod-classifier` 在本地对数值序列执行统计量、平稳性、趋势、季节性和周期性分析。TypeSafe 官方明确警告 Jev 1.13 不擅长数学、计数和数值精度，推荐把算术留在代码里。因此，不应把原始时间戳和采样值交给 Jev，让它取代 ADF/KPSS、STL、FFT 或本地检测算法。[Jev 1.13 已知限制](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

它可能有价值的窄场景是：在本地统计特征之外，还需要根据指标含义、标签、业务意图、风险等级或运行策略，在一个闭集里做语义判断。例如，从 `dynamics`、`kmeans`、`forecast`、`defer_to_local` 中选择一个候选，并在低 confidence 时交回本地策略。若输入只有数值特征且映射规则已经明确，直接用 Rust 规则或训练一个以历史模型表现为标签的监督式 selector，会比增加远程 Jev 依赖更简单、可复现。

## Jev 实际是什么

Jev 是 TypeSafe 的首个 “System One” 模型。一次调用提交一个 `state` 和一组带类型的 `questions`；模型不生成自由文本，而是返回代码可直接检查的结构化答案。一个请求里的问题共享同一份 state，但彼此独立并行求值。[官方介绍](https://docs.typesafe.ai/introduction) · [System One 概念](https://docs.typesafe.ai/concepts/system-one) · [State 说明](https://docs.typesafe.ai/concepts/state)

| primitive | 适用问题 | 输出 | 约束 |
| --- | --- | --- | --- |
| `Choice` | 从固定候选中选一个；分类、模型选择、路由 | `choice`、每个候选的 `probabilities`、`confidence` | 最多 255 个候选；候选须由调用方预先声明。[文档](https://docs.typesafe.ai/primitives/choice) |
| `Score` | 沿一个有序、文字描述的等级量表评分 | 概率加权的 `score`、各等级概率、`confidence` | 2–10 个等级；不是精确数值回归。[文档](https://docs.typesafe.ai/primitives/score) |
| `Noul` | 判断一个 yes/no 命题 | 命题为真的概率 `noul`，范围 0–1 | 不另带 `confidence`。[文档](https://docs.typesafe.ai/primitives/noul) |

“类型安全”只保证输出落在已声明的结构和候选空间内，不保证语义判断正确。官方文档说明校准是大量预测的群体频率属性，不保证某次答案正确；客户协议也明确服务可能产生不准确或错误的输出，客户必须独立评估。[AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) · [客户协议 §9.3](https://typesafe.ai/legal/mca)

它也不是 coding agent 或生成模型的替代品：不会聊天、写代码、解释理由或发现调用方未声明的新模型。TypeSafe 官方把 routing、classification、scoring、guardrails 列为适用场景，同时明确不能把 `jev-latest` 当作 coding agent 的主模型。[Jev with coding agents](https://docs.typesafe.ai/introduction/coding-agents)

## 输入、输出与推理接口

线上推理通过带 Bearer API key 的 HTTP 请求完成：

```http
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <TYPESAFE_API_KEY>
Content-Type: application/json
```

```json
{
  "model": "jev-1.13.0",
  "state": {
    "metric_semantics": "request latency for a user-facing API",
    "local_classification": "seasonal_with_trend",
    "data_quality": "sufficient",
    "operational_policy": "prefer explainable local methods"
  },
  "questions": {
    "detector": {
      "type": "choice",
      "instructions": "Which available detector best matches this state?",
      "criteria": {
        "baseline": "stable or irregular series where a robust local baseline is preferred",
        "kmeans": "fixed-resolution distance-based anomaly detection",
        "forecast": "trend-aware or seasonal-with-trend prediction",
        "defer_to_local": "evidence is insufficient or ambiguous"
      }
    }
  }
}
```

响应会返回实际使用的版本、同名 answer 和 token usage；`Choice` 的赢家是最高概率候选，概率覆盖所有候选且总和为 1。[完整 HTTP contract](https://docs.typesafe.ai/api)

`state` 可以是字符串、JSON object 或 array，但 Jev 当前只理解文本内容，不支持图像、音频或视频。当前模型每请求上下文上限为 64k token，同时 state 加最长单个问题上限为 32k；英文是主要训练语言，CJK 可用但官方称准确率较低。[State](https://docs.typesafe.ai/concepts/state) · [Models](https://docs.typesafe.ai/models)

## 训练方式与可定制性

TypeSafe 公开到的技术细节只到以下层级：新模型架构、并行 sampler，以及名为 RLCD（Reinforcement Learning for Calibrated Decisions）的后训练方法；优化目标是返回决策和校准概率，而非生成文本。[发布说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev) · [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)

官方没有发布 Jev 的权重、参数量、基座、训练数据配方、可复现实验或本地训练/推理流程。客户也不能上传数据做账号级 fine-tune 或 LoRA；所有账号使用相同权重，只能通过 state、instructions 和 criteria 注入领域信息。[Models：Customizing Jev](https://docs.typesafe.ai/models#customizing-jev)

因此，“训练 Jev 来替换本项目分类器”目前不可行；可做的是调用托管模型，或在 Jev 输出概率之上另外训练本项目自己的下游 selector。后者若使用 Jev 输出做训练数据，还需先核对客户协议对模型蒸馏、模仿输出及竞争产品的限制。[客户协议 §2.3](https://typesafe.ai/legal/mca)

## 官方仓库与包结构

官方没有 Jev 模型仓库。公开的相关实现均是客户端或兼容层：

| 仓库/包 | 当前发布 | 运行要求 | 内容与许可 |
| --- | --- | --- | --- |
| [`typesafe-sdk-python`](https://github.com/typesafe-ai/typesafe-sdk-python) / `typesafe-sdk` | [`v0.7.1`](https://github.com/typesafe-ai/typesafe-sdk-python/releases/tag/v0.7.1) | Python ≥3.10 | 同步/异步 HTTP client、Pydantic request/response types、retry、models API；MIT。[manifest](https://github.com/typesafe-ai/typesafe-sdk-python/blob/0ffd094c72ed9445223060b24ffd7a56aa781fb4/pyproject.toml) · [LICENSE](https://github.com/typesafe-ai/typesafe-sdk-python/blob/0ffd094c72ed9445223060b24ffd7a56aa781fb4/LICENSE) |
| [`typesafe-sdk-js`](https://github.com/typesafe-ai/typesafe-sdk-js) / `@typesafe-ai/sdk` | [`v0.6.0`](https://github.com/typesafe-ai/typesafe-sdk-js/releases/tag/v0.6.0) | Node.js ≥20 | ESM、CommonJS、TypeScript declarations；client、questions/types、retry、models resource；MIT。[manifest](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/package.json) · [source](https://github.com/typesafe-ai/typesafe-sdk-js/tree/66880ccded6cb642dc1809620c2b108c33730214/src) · [LICENSE](https://github.com/typesafe-ai/typesafe-sdk-js/blob/66880ccded6cb642dc1809620c2b108c33730214/LICENSE) |
| [`system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python) / `system-one-adapter` | [`v0.2.1`](https://github.com/typesafe-ai/system-one-adapter-python/releases/tag/v0.2.1) | Python ≥3.10 | 用 OpenAI/Anthropic/Gemini 等普通 LLM 模拟 System One API，供对比；它不是 Jev 权重或本地 Jev runtime；MIT。[README](https://github.com/typesafe-ai/system-one-adapter-python/blob/e1d4cc938204b22fc5a3c3aca7044072fe3f712d/README.md) |

官方 SDK 列表只有 Python 和 JavaScript/TypeScript，但文档明确任何语言都可直接调用 HTTP API；没有官方 Go 或 Rust SDK。[SDK 文档](https://docs.typesafe.ai/sdk)

## 对分类、模型选择、路由和打分的适配度

| 需求 | 适配度 | 判断 |
| --- | --- | --- |
| 文本/业务语义分类 | 高 | `Choice` 就是闭集分类器；官方提供了 75 类文档分类示例，并通过 confidence 决定精确分类或降级到粗分类。[官方 cookbook](https://docs.typesafe.ai/cookbooks/classification_using_confidence) |
| 在预定义模型/处理器之间路由 | 高 | 官方 intent-routing 模式直接把请求路由到确定性代码、专用 LLM 或人工；最终控制流仍在代码中。[Intent routing](https://docs.typesafe.ai/patterns/intent-routing) |
| 定性风险/复杂度/质量打分 | 中高 | 适合有文字锚点的等级量表；复杂评分应拆成独立维度后在代码中加权。[Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring) |
| 原始数值时间序列分类 | 低 | 官方明确称数学、计数、数值表示和精确量值是弱项；应在代码里完成。[Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13#math-and-numbers) |
| 自动发现新候选或生成模型配置 | 不适合 | 输出只能在调用方预声明的 Choice/Score/Noul 空间中，不生成字符串或代码。[System One](https://docs.typesafe.ai/concepts/system-one) |
| 替代本地模型训练/推理 | 不适合 | 托管 API、无公开权重、无客户 fine-tune/LoRA。[Models](https://docs.typesafe.ai/models) |

## 与当前 Alert4ML 的具体关系

尽管仓库位于 Go 工作区路径，项目接口说明已记录 Go backend 被 Rust `rsod-backend` 替换。当前调用链是 TypeScript Grafana 前端 → Rust plugin backend → Rust 算法 crates；Jev 不应接在浏览器前端，也没有新的 Go 层。[本项目接口说明](interfaces/README.md)

现有 [`rsod-classifier`](../rsod/crates/rsod-classifier/src/pipeline.rs) 在本地从数值序列计算统计量、平稳性、趋势、季节性和周期性，输出 `SeriesCharacteristic` 和规则式 confidence。检测器由产品配置在 Dynamics、K-means 与 Forecast 三者中显式选择，不存在自动算法路由层。

这也是为什么“直接用 Jev 替换分类器”不划算：它会把低延迟、离线、可复现的数值判断改成依赖外网和供应商 API 的文本判断，而且恰好落在 Jev 官方列出的数值弱项上。Jev 更合理的角色是一个**可选的语义 policy router**：消费本地算法已经算好的离散/文字化特征，加上指标语义和业务策略，然后在已注册 engine 闭集里给出第二意见。

推荐的边界如下：

```text
raw series
    │
    ├── 本地 rsod-classifier ──────────────────────────────┐
    │                                                     │
    └── 精简、离散化的事实 + metric 语义 ──> Jev Choice ──┤
                                                          ▼
                                         本地 policy / confidence gate
                                                          │
                                    dynamics | kmeans | forecast | fallback
```

Jev 只提供证据；Rust policy 决定是否采纳，并拥有超时、回退、审计和最终副作用。这里讨论的是检测算法路由，不包含 Prometheus rule / PromQL 推荐或自动建规则；该能力仍受 [ADR 0001](adr/0001-park-prometheus-static-alert-suggestions.md) 约束并保持不存在。

## 最小集成方案

建议分阶段验证，而不是立即进入 Grafana Alerting 主链：

1. **离线 benchmark**：先用 Python SDK 读取固定历史样本，不改插件。ground truth 不是现有规则的输出，而是各候选 detector 在后续窗口上的真实表现；衡量 selector regret、分类准确率、概率校准、abstain 后覆盖率、成本和延迟。
2. **Rust backend shadow mode**：若离线结果有价值，在 `rsod-backend` 增加一个小型 HTTP client。该 crate 已依赖 `reqwest`；不要把网络调用或 API secret 放进 `rsod-classifier`、`rsod-core` 或浏览器 bundle。用独立的 `typesafeApiKey` 存入 Grafana `secureJsonData`，不要复用现有 Grafana `apiToken`。
3. **只发送必要摘要**：本地先计算数值特征并转成清晰的命名字段/区间，例如 `stationarity=non_stationary`、`seasonality=strong`、`history_quality=sufficient`；不要发送整段原始时间序列，也不要要求 Jev 做均值、计数、日期差或阈值算术。
4. **固定 contract 与版本**：生产实验固定 `jev-1.13.0`，不要用会随发布移动的 `jev-latest`。本地严格校验 answer id、type、候选集合、概率范围与总和；记录响应中的实际 model 版本。[模型 alias 说明](https://docs.typesafe.ai/models#aliases)
5. **强制回退**：超时、401/422、429/529、响应校验失败、低 confidence 或候选为 `defer_to_local` 时，一律使用现有 `decide()`。官方 HTTP contract 要求对 429/529 backoff，但 Grafana 告警查询不应为远程 selector 长时间重试。[API errors](https://docs.typesafe.ai/api#errors)
6. **先观察、后授权**：shadow 阶段只记录 Jev 与本地选择的差异，不改变 detector。只有在自有历史数据上证明显著降低模型选择 regret，并通过中文输入、服务故障和负载测试后，才考虑让它处理低风险分支。

已移除的 Funnel 曾把“是否触发 L2 → 分类 → 路由 → 执行”封装在同步算法调用里。若未来接入远程 Jev，应在异步 `rsod-backend` 编排层调用，再把已校验的可选 `RoutingOverride` 传入纯算法层；不要在同步 detector 内阻塞 HTTP，也不要让算法 crate 持有凭据。

## 成熟度、维护状态和主要限制

风险等级为 **高（适合实验，不宜成为告警唯一判定路径）**：

- Jev 于 2026-09-15 首次公开，发布说明称其为 early access；截至本评估日只有一周公开历史。[发布说明](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- 当前稳定版本是 `jev-1.13.0`，官方 SDK 仍是 0.x，但 Python SDK 在 9 月 21 日、adapter 在 9 月 22 日仍有发布，表明项目活跃而非停更。[Models](https://docs.typesafe.ai/models) · [Python releases](https://github.com/typesafe-ai/typesafe-sdk-python/releases) · [adapter releases](https://github.com/typesafe-ai/system-one-adapter-python/releases)
- 官方价格为每百万输入 token $0.042，输出 token 免费；公开限额为 250k token/s、1,200 requests/min，但官方明确说限额仍会动态调整且可不预告。这些不是 Alert4ML 可依赖的本地 SLA。[Models](https://docs.typesafe.ai/models)
- 已知弱点包括字面化理解、数学/数值/日期比较、多跳间接推断、无关长 state、state 内对抗性内容、instructions 与 criteria 冲突，以及不同 primitive 之间不保证直觉上的概率恒等关系。[Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- 英文效果最佳，中文和其他 CJK 输入必须单独评估；本项目若用中文 metric 描述或业务策略，不能外推英文 benchmark。[Models：Language support](https://docs.typesafe.ai/models#language-support)
- 服务条款允许 TypeSafe 为提供服务处理输入，并可长期处理派生 telemetry；官方称不会未经同意用客户数据修改模型权重，企业客户可询问 ZDR。把监控指标、标签或业务上下文发送给第三方前，仍需完成数据分类和合规审查。[客户协议 §4](https://typesafe.ai/legal/mca) · [Legal / ZDR](https://docs.typesafe.ai/legal)
- API 服务可能更新并造成客户应用不兼容；协议只承诺对其认为有重大不利影响的更新作商业上合理的预告。[客户协议 §2.5](https://typesafe.ai/legal/mca)

## 决策建议

**可以用，但用途要收窄为“远程、闭集、语义型的第二层路由器”，不能把它当作开源本地模型，也不应替代当前数值分类器。**

若模型选择主要由季节性、趋势、偏度、缺失率和历史回测误差决定，继续使用本地确定性逻辑或训练专门的监督式 meta-model；Jev 不会带来足以抵消网络、合规和供应商依赖的优势。若选择还依赖指标业务语义、用户意图、安全/成本策略等难以手写的文字规则，则值得做 shadow POC。通过门槛应由本项目自有数据上的质量、校准、延迟、成本和故障回退结果决定，而不是 TypeSafe 的宣传 benchmark。
