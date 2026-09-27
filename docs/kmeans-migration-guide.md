# Funnel 到 K-means 迁移与回滚指南

本文面向运维人员、Dashboard 维护者和 Grafana Alerting 规则维护者。算法原理见
[K-means 算法说明](kmeans_algorithm.md)，Netdata 参考实现与差异见
[研究记录](netdata-kmeans-research.md)，验证与容量验收见
[K-means 验证与容量手册](kmeans-validation-and-capacity.md)。

## 迁移原则

- `funnel` 不会被后端静默解释为 `kmeans`。未迁移的旧查询会返回
  `Funnel L1 has been removed; migrate this query to K-means`。
- 迁移会保留上游数据源查询、targets、序列显示标签和输出模式，但会丢弃
  Funnel 专属参数，创建全新的 K-means 配置和模型。
- Grafana 的显示步长仍可随时间范围和面板宽度变化；保存的
  `detectIntervalMs` 才是训练和评分步长。两者不能混用。
- 不自动修改已保存 Dashboard、Library Panel 或 Alerting rule，也不自动删除
  Funnel/K-means 的数据库记录。

## 上线前清单

1. 盘点所有保存内容中 `"detectType":"funnel"` 的查询，并记录 Dashboard UID、
   panel ID、rule UID、查询 refId、上游数据源 UID 和负责人。Library Panel 和告警
   规则必须单独盘点，不能只搜索 Dashboard JSON。
2. 导出待迁移 Dashboard 和告警规则 JSON，保存当前插件版本、基线提交
   `a1e5cbb`、告警状态和通知策略。这些文件是回滚依据。
3. 确认上游数据源能在拟定的固定步长返回真实数据。默认步长为 60 秒；如果上游
   只有 5 分钟数据，应显式使用 `detectIntervalMs: 300000`，不能把 5 分钟数据
   插值成 1 分钟训练数据。
4. 检查查询中的 `$__interval`、`$__interval_ms` 和 `$__rate_interval`。K-means
   路径会在模板展开前把它们固定到检测步长，但查询中显式写出的窗口（例如
   `rate(x[5m])`）保持不变，并进入模型身份。
5. 使用生产代表性历史数据完成因果回放并预先约定准确率、延迟、覆盖率和资源
   门槛。仓库自动化测试不等同于生产准确率证明。

## 单个查询的迁移步骤

1. 先复制 Dashboard 或告警规则，在副本中操作。不要直接覆盖唯一的线上规则。
2. 打开旧查询。编辑器会显示 **Funnel has been removed**，点击
   **Migrate to K-means**。
3. 核对上游数据源和原查询、targets、`seriesLabel`、`outputMode` 是否保留。
4. 设置固定检测步长。新查询默认 `detectIntervalMs: 60000`，K-means 编辑器不提供
   `Auto (panel)`。修改步长会产生新的模型身份并重新训练。
5. 使用默认 60 小时历史，除非回放证明需要其他值。默认参数需要 18 个、间隔
   3 小时的 6 小时模型；历史过短会保持 `WarmingUp`。
6. 先用 `visualization` 输出观察数据质量、分数和模型状态。训练不足或历史缺失是
   “未作判定”，不是正常值 0。
7. 告警规则使用独立的 `alert_signal` 查询。Grafana 表达式使用
   `Reduce Last(<refId>)` 后接 `Threshold > 0.5`，并明确配置 NoData、Error 和
   `for` 策略。不要让多 frame 的 `visualization` 查询直接进入告警表达式。
8. 等待达到预期模型成熟度，执行同一时间段的 replay/影子比较，再保存并启用
   新规则。先迁移少量代表性序列，观察一个完整业务周期后再扩大范围。
9. 新规则稳定后再停用旧规则，避免两个规则同时发送通知。保留导出的旧 JSON，
   直到回滚窗口结束。

## 推荐配置示例

以下片段展示线上的默认算法契约；`targets`、数据源 UID 和唯一键必须替换为实际值。
`anomalyThreshold: 99` 是距离分数阈值，不是 99% 置信度或百分位。

```json
{
  "supportDetect": "machine_learning",
  "detectType": "kmeans",
  "detectIntervalMs": 60000,
  "historyTimeRange": { "durationMs": 216000000 },
  "outputMode": "alert_signal",
  "hyperParams": {
    "trainWindowSecs": 21600,
    "retrainEverySecs": 10800,
    "maxModels": 18,
    "minModelsForAlert": 18,
    "anomalyThreshold": 99,
    "minTrainVectors": 120,
    "maxTrainVectors": 1440,
    "maxIterations": 1000,
    "seed": 82795246472787,
    "persistModel": true
  },
  "targets": [{ "refId": "A", "datasource": { "uid": "replace-me" } }],
  "uniqueKeys": {
    "dashboardUid": "replace-me",
    "panelId": 1,
    "seriesRefId": "A"
  }
}
```

前端和 Rust 默认 seed 使用相同的十六进制值 `0x4b4d45414e53`；JSON 示例采用其
十进制表示。不要为了缩短冷启动而直接把 `minModelsForAlert` 降为 1；仓库 E2E
测试这样配置只为缩短测试时间，不是生产建议。

## 行为变化

| 项目       | Funnel 旧行为                     | K-means 新行为                                               |
| ---------- | --------------------------------- | ------------------------------------------------------------ |
| 检测入口   | `detectType: funnel`              | `detectType: kmeans`                                         |
| 时间分辨率 | 历史可能随 `maxDataPoints` 变粗   | 训练和评分共用保存的固定步长                                 |
| 模型       | L1/L2、画像和统计路由             | 每序列独立的 `k=2` 模型集合                                  |
| 输出解释   | baseline/bounds 等 Funnel 语义    | 距离分数、异常位、有效性和 readiness                         |
| 未就绪     | 旧路径可能难以区分未知与正常      | `WarmingUp`/无效点；AlertSignal 输出 null                    |
| 显示缩放   | 可能影响算法输入                  | 只影响评分后的显示聚合                                       |
| 旧参数     | trend、内外带、bucket、profile 等 | 不映射；使用新的 K-means 默认参数                            |
| 模型复用   | Funnel 模型身份                   | 步长、完整序列身份、预处理版本、训练配置和查询共同决定新身份 |

## 回滚步骤

回滚包含“插件版本”和“保存内容”两部分，仅回滚二进制不会把已保存的 K-means
查询自动变回 Funnel。

1. 暂停新 K-means 告警或通知，记录发生回滚的时间和原因。
2. 恢复上线前导出的 Dashboard/Library Panel/Alerting rule JSON，确认查询重新指向
   旧的 `detectType: funnel` 及原参数。
3. 部署基线 `a1e5cbb` 对应的插件构建，并完成 Grafana 插件健康检查。
4. 在非通知模式下验证旧查询能出数，再按原顺序恢复规则和通知。
5. 保留回滚期间的 K-means 查询结果、日志和模型存储用于复盘。当前没有面向用户的
   K-means 模型清理命令；K-means 使用独立的 `kmeans:v1:` 存储前缀，旧版本不会
   把它误当作 Funnel 模型。是否清理必须另开受控的数据迁移任务，不能在回滚中直接
   删除数据库记录。

## 已知限制与运维提示

- K-means 使用差分特征，可能弱化绝对水平、持续阶跃、缓慢漂移和“预期事件未发生”
  的信号；不能把它当作所有统计基线告警的等价替代。
- 默认模型达到 18 个之前告警输出为 null；历史不足、缺口或分辨率不匹配也不能
  当作正常。NoData 策略必须由规则所有者明确选择。
- null、NaN、Inf 和缺失时间格不填 0；一个缺口会使重新积累九个连续原始样本前
  的特征无效。
- 上游实际只保存粗粒度数据时，插件无法恢复被聚合掉的信息。时间戳看似落在细网格
  上也不等同于拥有原始精度。
- 模型按 series labels 独立训练。标签基数增加会线性增加取数、训练、内存和持久化
  压力；容量边界及测量方法见验证与容量手册。
- 模型快照按 3 小时 anchor 使用不可变键保存。当前存储实现保留旧 anchor，未实现
  自动保留期清理；长期运行前必须监控模型表增长。
- 历史缓存的持久化仅在 Postgres 模式启用；trial 模式为内存 SQLite，重启后需要
  重新取数/训练。缓存和模型持久化失败会降级，不能据此承诺无冷启动。
- 节点级 NAR/事件、DAR 和逐点不可变异常记录不在本次首版范围内。
