# Google SRE 告警基线与企业私有策略实例

> 目的：将 Google SRE 的通用、可审计告警原则作为 Alert4ML 的**策略基线**；企业自己的业务语义、风险偏好和事故经验以受控的私有策略实例叠加。本文不是对 Google 实践的逐字复刻，也不把任一具体阈值当作通用真理。

## 结论

AI 应负责把 SRE 的自然语言经验转为结构化场景、发现缺失信息、从候选策略中排序并解释；**确定性策略引擎**负责验证前置条件、选择算法、计算阈值、输出路由和降级方案。任何 `page` 都必须由人明确批准，且应可回放、可解释、可撤销。

Google 的核心思想可归为一句话：面向用户可见的服务目标（SLO/SLI）告警；只有需要立即人工行动的重大症状才打扰值班人员，其余进入工单或仪表盘。Google 明确区分 page、ticket 和日志，并强调不要要求人从告警中再判断是否需要行动。[Production Services Best Practices](https://sre.google/sre-book/service-best-practices/)；[Introduction](https://sre.google/sre-book/introduction/)

## 讨论记录与设计共识

### 目标

将每个企业 SRE 团队掌握的告警经验，沉淀成可复用、可程序化且可审计的资产；AI 根据企业自己的场景，在 Alert4ML 已支持的算法和策略中给出推荐，但不替代 SRE 对生产 page 的责任与审批。

### 已达成的原则

1. **经验不是算法对照表。** “CPU 使用率使用算法 A”不可迁移；可迁移的是告警意图、用户影响、数据特征、处置动作、限制条件和复盘证据。
2. **先选择告警语义，再选择算法。** 错误预算耗尽、容量即将耗尽、周期性偏离、突发异常和预期事件缺失分别是不同的问题。算法只提供证据，不能单独决定 page。
3. **Google SRE 是基线，不是硬编码参数集。** 其 page/ticket/info 分级、症状优先、SLO、错误预算、低流量保护、可行动性和告警风暴控制应成为通用约束；具体窗口、burn rate、最小样本量和路由须由企业实例覆盖并经验证。
4. **通用模板公开，企业实例私有。** 模板共享结构、适用条件、反例、降级路径和解释；私有实例保存指标、阈值、业务日历、事故引用和组织路由。跨企业共享时只输出去标识化的经验。
5. **AI 负责理解和推荐，策略引擎负责决定和验证。** AI 可进行场景访谈、补全 Profile、解释候选方案和总结反馈；确定性模块必须验证数据质量、SLO/流量护栏、算法适用性及严重性规则。
6. **新增策略先观察、后通知。** 先 shadow 和回测，再由策略 owner 批准进入 ticket，最后才可提升为 page；任何版本均须可回放和回滚。

### 系统职责图

```text
SRE 描述 / 指标目录 / 历史数据 / 变更与事故记录
                         │
                         ▼
                AI 场景访谈与结构化
                         │ AlertProfile（人工审核）
                         ▼
Google SRE 基线 ──► 通用模板 ──► 企业私有策略实例
                         │
                         ▼
        确定性策略引擎：资格检查、候选排序、算法、
        告警级别、路由、抑制、降级与可解释决策
                         │
                         ▼
            shadow / 回测 ──► SRE 审批 ──► 受控启用
                         │                         │
                         └──── 人工反馈、复盘、版本演进 ┘
```

### Alert4ML 中算法的定位

| 告警问题 | 策略优先级 | 可用算法/机制 | 重要护栏 |
|---|---|---|---|
| 高流量服务的可用性、正确性或延迟 SLO 风险 | 首选 | SLI 比率、错误预算、多窗口 burn rate | 最小请求数、最小坏事件数、page 必须有行动方案 |
| 正常值随工作日/周周期变化的用户症状 | 次选 | `Dynamics`、`Funnel`、动态基线 | 足够的完整历史周期、稳定采样、变更期处理 |
| 平稳连续信号的突发偏离 | 次选 | `outlier`、MAD/IQR/EVT | 防抖、持续时间、不能将单点越界直接 page |
| 有上限的资源即将耗尽 | 首选于容量场景 | `forecast` / Perpetual 的耗尽时间预测 | 上限与可行动窗口必须明确；模型失败时保守降级 |
| 心跳、批任务、业务事件未按预期出现 | 首选 | 缺失/新鲜度规则 | 用业务期望表达，不套用连续数值的季节性模型 |
| CPU、单 Pod、队列等原因信号 | 诊断优先 | 上述算法可作为证据 | 默认不独立 page；关联到服务症状或容量风险 |

### 最小可行落地顺序

1. 建立指标语义合同：`service`、`owner`、指标类型/单位、方向、分母、用户影响、SLO、runbook 和业务日历。
2. 将现有 `rsod-classifier` 的序列画像扩展为策略输入：数据质量、缺失率、采样稳定性、历史覆盖周期数、趋势、季节性和突变。
3. 引入一个独立的策略模块，输入为 `AlertProfile` 与序列画像，输出完整 `PolicyDecision`；不要让前端或算法 crate 分散承担策略选择。
4. 先实现少量高价值通用模板：`slo_burn_multiwindow`、`seasonal_symptom_baseline`、`forecasted_capacity_exhaustion`、`low_traffic_failure_guarded`、`missing_expected_event`。
5. 在 Grafana 展示推荐、证据、风险、降级和“为什么不是 page”；允许 SRE 覆盖、锁定及创建私有实例。
6. 将批准、覆盖、告警结果和复盘关联记录为反馈；初期仅用反馈改善候选排序，不让学习系统绕过人工审批。

## 应成为产品默认值的通用策略

| 原则 | Google 一方依据 | Alert4ML 的通用策略基线 |
|---|---|---|
| 从用户体验和服务目标出发 | Google 建议 paging 与威胁服务 SLO 的症状对齐；黑盒监控观察用户可见行为。[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) | 每个可分页策略必须关联一个 `service`、`SLI`、`SLO` 和用户影响说明。机器 CPU、单 Pod 重启等原因信号默认不可直接 page，只能作为诊断或相关性信号。 |
| Page 必须立即可行动 | Google 要求所有告警立即可行动；page 后应存在系统不能自动完成的即时人工动作。[On-Call](https://sre.google/workbook/on-call/) | `page` 策略必须有 owner、runbook、明确动作、影响范围和升级目标；缺任一项则降为 `ticket` 或 `info`。 |
| 通知分级而不是“所有异常都告警” | Google 将 page 定义为现在必须行动、ticket 定义为数天内行动、日志留作事后分析。[Production Services Best Practices](https://sre.google/sre-book/service-best-practices/) | 决策输出必须包含 `severity: page | ticket | info`。异常检测算法的原始分数不是 severity；它需要结合影响、SLO 和剩余处置时间。 |
| SLO 与错误预算是分页的共同尺度 | Google 用错误预算将可靠性目标转成可量化的风险，并以 burn rate 说明耗尽速度。[Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | 对可定义 SLI 的服务，优先产生 `burn_rate`、预算消耗比例和预计耗尽时间；固定阈值或 ML 结果可作为 SLI 的输入/补充，不能隐去此业务尺度。 |
| 多窗口、多 burn-rate 覆盖快慢故障 | Google 建议使用多个 burn rate 与窗口，兼顾高召回和快速恢复；文中给出“1 小时消耗 2%、6 小时消耗 5%”作为两个 paging 起点、“3 天消耗 10%”作为 ticket 起点，并明确这些数字需按服务和 page 负荷调整。[Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | 提供参数化模板，而非硬编码数值：每个 burn-rate 规则可配一对短/长窗口，二者同时满足才 page（降低瞬态误报）；较慢且重要的预算消耗建 ticket。将 Google 数值标记为 `starter`，允许企业基于误报成本与值班能力覆盖。 |
| 比率要加体量/低流量保护 | Google 展示了低流量服务中单次失败可能产生极高 burn rate 的问题。[Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | 所有 error ratio/burn-rate 策略须声明最小请求数或最小坏事件数；不满足时输出 `insufficient_traffic`，不自动 page。低频业务使用更长窗口、绝对坏事件门槛、合成探针或人工定义的业务规则。 |
| 防抖与恢复行为是规则的一部分 | Google 的通用时序告警规则可要求条件持续一段最短时间，通常至少两个评估周期，以避免漏采集导致误报；规则实例也同时要求 error ratio 与错误绝对速率。Google 对 SLO 分页则比较 burn-rate 的检测和恢复时间，不建议只用一个 `for` 来表达不同严重度。[Practical Alerting from Time-Series Data](https://sre.google/sre-book/practical-alerting/)；[Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) | 策略模型须包含评估间隔、最小样本量、恢复条件和缺失数据语义。普通异常策略可使用 `for`；SLO page 优先用多窗口 burn-rate 条件。不要将单点的 outlier/forecast 越界直接转成 page。 |
| 聚合、去重、抑制告警风暴 | Google 的 Alertmanager 能按标签抑制、去重、聚合；并指出单一异常可产生多个告警，应调节 fan-out。[Practical Alerting from Time-Series Data](https://sre.google/sre-book/practical-alerting/)；[Being On-Call](https://sre.google/sre-book/being-on-call/) | 每个策略必须定义 group key、父子抑制关系、去重窗口和通知上限。症状 alert 优先；相同事故中的原因 alert 附着在事件上下文中。 |
| 健康告警应聚合到服务层 | Google 指出单机失败在大规模系统中通常过于嘈杂，应该对高层服务目标告警，同时保留组件粒度供排障。[Practical Alerting from Time-Series Data](https://sre.google/sre-book/practical-alerting/) | 算法的默认输入应支持按 `service/region/criticality` 聚合的 SLI；高基数实例指标用作 root-cause evidence，而非逐实例 page。 |
| 告警信息必须可诊断 | Google 的规则在触发通知中填入 job、告警名和触发值等上下文；on-call 指南要求 page 链向相关监控控制台。[Practical Alerting from Time-Series Data](https://sre.google/sre-book/practical-alerting/)；[On-Call](https://sre.google/workbook/on-call/) | 每个推荐必须生成解释包：输入 SLI/SLO、算法版本、特征与数据质量、当前值/阈值或预测区间、触发窗口、相关仪表盘、runbook 与推荐动作。 |

## AI、通用策略与私有实例的职责边界

```text
SRE 叙述 + 指标元数据 + 历史序列 + 事件/变更记录
                         │
                    AI 场景访谈
                         │
             结构化 AlertProfile（可审阅）
                         │
        Google-SRE 通用基线 + 企业私有策略实例
                         │
       确定性 Policy Engine：候选、约束、算法、降级
                         │
       SRE 批准 → shadow 回测 → 受控启用/反馈/版本化
```

### 1. AI 不能替代的字段

AI 可以从描述中提取候选值，但提交前必须让 SRE 审核下列字段：

```yaml
service: checkout
sli:
  kind: availability | latency | correctness | freshness
  good_event: "..."
  total_event: "..."
slo:
  objective: 99.9
  period: 30d
alert_intent: user_harm | imminent_exhaustion | capacity_exhaustion | task_missing
user_impact: "支付请求失败"
severity_and_action:
  page: "立即切流或回滚"        # 必须是人现在能做的事
  ticket: "下一工作日分析"
owner: payments-oncall
runbook: "..."
```

`SLO` 不存在时，系统可以推荐动态基线、预测或异常检测，但结果默认是 `ticket/info`；升级为 `page` 必须声明等价的用户伤害和紧急行动依据。

### 2. 通用策略模板：公开、版本化、可移植

模板只保存没有企业机密的决策结构，例如：

- `slo_burn_multiwindow`：适用于有足量流量的可用性/正确性 SLI；算法为比率与 burn-rate 计算，不用无监督异常检测替代。
- `seasonal_symptom_baseline`：适用于正常值受日/周规律影响的用户症状；使用 Baseline/Dynamics/Funnel 等候选算法，但必须先满足足够历史、稳定采样和缺失率约束。
- `forecasted_capacity_exhaustion`：适用于有明确上限、可操作的容量指标；用 Forecast/Perpetual 估计耗尽时间，page 的前提是耗尽发生在可行动窗口内。
- `low_traffic_failure_guarded`：低流量错误率的最小样本、绝对错误数和长窗口保护；通常优先 ticket。
- `missing_expected_event`：心跳、批任务或关键业务事件“应出现却未出现”；这是业务期望规则，不应硬套连续数值的季节性算法。

模板必须带有 `eligibility`（历史长度、采样稳定性、缺失率、分母下限）、`not_for`、`fallback` 和 `explanation` 字段。算法选择应是模板内部的可替换实现，不是模板的身份。

### 3. 企业私有策略实例：继承而非复制

私有实例引用一个基线模板，并只覆盖本企业独有的语义和证据：

```yaml
id: payments-api-availability-v3
inherits: slo_burn_multiwindow@1
scope: {service: payments-api, region: cn-east}
overrides:
  slo: {objective: 99.95, period: 28d}
  traffic_guard: {min_requests: 500, min_bad_events: 5}
  notification:
    group_by: [service, region]
    inhibit_when: ["payments-api-global-outage"]
  calendar: {release_windows: "private calendar reference"}
evidence:
  incident_refs: [INC-2026-041]
  backtest_id: bt-2026-08-17
  accepted_by: sre-team
```

私有实例还应记录：创建/审批人、有效期、算法和模型版本、所用历史范围、回测结果、每次人工覆盖、事件/变更关联、回滚目标。公开共享时，只提炼去标识化的模板改进和“为什么”，不导出指标名、阈值、业务日历、事故细节或原始序列。

## 策略选择的硬性护栏

1. **症状优先，但保留原因证据。** 将 CPU、队列、单实例错误等标记为 `cause_signal`；它们可提高诊断置信度或生成 ticket，但不得绕开用户影响/SLO 的 page 门槛。Google 的告警哲学强调应对用户可见症状而非每个潜在根因进行打扰。[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
2. **Pager 关键路径保持简单。** Google 建议避免在 pager 判定中使用会自动学习阈值或推断因果的“magic”系统；复杂度放在离线评估、容量预测和诊断层。[Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) 因此 AI/ML 不能成为未审计的 page 决定者。
3. **将数据质量视为安全条件。** 季节性/预测算法需要检查采样规律、历史覆盖周期数、缺失率、重置/口径变更和异常值比例；失败时返回明确原因并使用保守 fallback。
4. **将算法输出与告警决策分离。** `anomaly_score`、预测区间、趋势和置信度是证据；`severity` 由 SLO 影响、持续时间、流量和人工行动窗口决定。
5. **先 shadow，后通知。** 新策略或模型版本先产生观察事件和回测报告，不进入 pager；比较检出、误报、漏报、恢复时间、page 每班次数和预算消耗解释的一致性。Google 对 SLO alert 比较 precision、recall、detection time 和 reset time。[Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/)
6. **每次 page 都要能复盘。** 保存可复算快照和策略版本；若无法回答“哪项 SLO、消耗了多少预算、为什么现在需要人做什么”，该 page 不应上线。

## 建议的最小决策输出

```yaml
decision:
  template: slo_burn_multiwindow@1
  algorithm: burn_rate_ratio
  candidates_considered: [burn_rate_ratio, seasonal_baseline]
  severity: page
  trigger:
    burn_rate_pair: high_urgency
    long_window: {window: 1h, budget_spend: 0.02}
    short_window: {window: "long_window / 12", budget_spend: 0.02}
    condition: both_windows_true_at_same_burn_rate
  guards:
    min_requests: 500
    min_bad_events: 5
    evaluation_interval: 1m
    missing_data: no_data
  routing:
    group_by: [service, region]
    runbook: "..."
  explanation:
    slo_impact: "..."
    recommended_action: "..."
    confidence: high
  fallback: ticket_on_insufficient_traffic
```

这里的窗口和预算消耗只是可配置示意；Google Workbook 所列数值是合理起点而非适用于所有团队的默认 SLA。任何企业私有覆盖都应有书面理由、回测证据和到期复审。

## 一方资料清单

- [Google SRE Workbook: Alerting on SLOs](https://sre.google/workbook/alerting-on-slos/) — 错误预算、burn rate、多窗口、低流量限制。
- [Google SRE Book: Service Level Objectives](https://sre.google/sre-book/service-level-objectives/) — 从用户关心的服务行为定义精确 SLI 与可驱动正确行动的 SLO。
- [Google SRE Book: Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/) — page/ticket 定义、黑盒与白盒、症状/根因告警哲学。
- [Google SRE Book: Practical Alerting from Time-Series Data](https://sre.google/sre-book/practical-alerting/) — `for` 持续时间、双条件例子、通知上下文、抑制/去重/聚合。
- [Google SRE Workbook: Being On-Call](https://sre.google/workbook/on-call/) — 可行动 page、告警疲劳、控制台链接。
- [Google SRE Book: Production Services Best Practices](https://sre.google/sre-book/service-best-practices/) — page/ticket/log 输出分级、错误预算。
- [Google SRE Book: Being On-Call](https://sre.google/sre-book/being-on-call/) — 页面的症状对齐、分组和健康的 on-call 负荷。
