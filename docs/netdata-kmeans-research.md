# Netdata k-means 实现核对

研究日期：2026-09-27。目的：为本项目替换 Funnel L1 提供算法依据；本文件仅记录研究，不实现替换。

源码固定为 Netdata 提交 [`4464a9ed71c8a7351b540ce4ca207e1418faffea`](https://github.com/netdata/netdata/commit/4464a9ed71c8a7351b540ce4ca207e1418faffea)，提交时间 2026-09-26。下文源码链接均固定版本，避免随 master 变化。

## 1. 参考来源与优先级

- [用户提供的总体说明](https://learn.netdata.cloud/docs/netdata-ai/anomaly-detection)：说明按指标独立建模、模型共识和异常位/异常率的产品结构。
- [用户提供的准确性分析](https://learn.netdata.cloud/docs/netdata-ai/anomaly-detection/ml-accuracy)：说明目标设计，但其中特征构造、距离公式和统计保证与源码存在差异。
- 下文以官方源码的实际执行逻辑作为移植依据。不能把说明页中的推导直接当成可兑现的精度指标。

## 2. 六维向量的真实构造

真实顺序为 `diff → smooth → lag`，不是把一维差分、一维平滑和四维原始滞后值拼接。默认 `diff_n=1`、`lag_n=5`。

对按时间递增的原序列定义：

```text
d_t = x_t - x_(t-1)
u_t = (d_t + d_(t-1) + ... + d_(t-s+1)) / s
z_t = [u_(t-5), u_(t-4), ..., u_t]
```

六维都是同一个“差分后再平滑”序列的相邻值。训练和预测使用同一变换。训练从可用位置提取多个向量，预测只提取最近一个。

源码：[ml_features.cc L29–109](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_features.cc#L29)。默认值：[ml_config.cc L127–141](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_config.cc#L127)。

平滑长度的实际选择是：指标所在 chart 的 `update_every` 大于全局 `nd_profile.update_every` 时为 1；否则取 `max(max_samples_to_smooth, 1)`，默认 3。因此不能把条件写死为“只有一秒指标才平滑”。有效原始样本长度为 `diff_n + s + lag_n`，默认平滑情形为 9，非平滑情形为 7；这些是样本数量，不是固定秒数。

源码：[ml.cc L184–192](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L184)、[预测缓存 L1116–1183](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1116)。预测缓存填满的那个调用仍直接返回 false，下一次调用才进入评分。

含义：默认差分会去掉长期绝对水平，算法主要识别局部变化模式。不能保证持续偏高、平坦故障、缓慢漂移或“应该发生但没有发生”的周期事件都会被识别。这是特征公式推导出的限制，需要在本项目测试数据中专门验证。

## 3. k-means 与距离评分

训练调用 `dlib::pick_initial_centers(2, ...)` 再调用 `dlib::find_clusters_using_kmeans(...)`。默认最大迭代次数为 1000。这里确认的是 Netdata 调用方式，尚未固定并审计其 dlib 依赖版本内部的中心初始化细节；不要声称已验证与某个 Rust k-means 库完全等价。

两个簇都是用于描述训练数据的几何结构，没有“一个正常簇、一个异常簇”的标签。

对每个训练/预测向量 z，实际评分距离是到所有中心的欧氏距离均值：

```text
D(z) = (||z - c0||₂ + ||z - c1||₂) / 2
D_min = min(D(z_i)) over emitted training vectors
D_max = max(D(z_i)) over emitted training vectors
score(z) = min(100, 100 * abs((D(z) - D_min) / (D_max - D_min)))
```

当 `D_max == D_min` 时源码直接返回 0。注意绝对值 `abs`：距离低于训练最小值时也会产生正分数。这与常见的 `clamp((D-min)/(max-min), 0, 1)` 不等价。训练使用最近中心的 k-means 优化，不代表异常评分也使用最近中心。

源码：[ml_kmeans.cc L46–102](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_kmeans.cc#L46)。

`99` 是上述 min–max 距离分数阈值，不是训练分布的第 99 百分位；没有分位数校准，因此也不能解释为单模型 1% 误报概率。文档使用 nearest center/percentile 的表述不能作为实现规范。

## 4. 多模型共识与冷启动

预测遍历当前内存中的模型，有任意一个有限分数 `< 100 * threshold` 就返回 false；默认 threshold 为 0.99，因此等于 99 会投异常票。所有参与的有效模型都达到阈值且至少有一张异常票，才返回 true。

这不是强制等待 18 个模型齐备：只有 1 个已训练模型时也能判异常；没有模型时返回 false。NaN 分数被跳过，并不是自动异常票。移植时建议显式输出模型不足/评分失败状态，而非把这些状态解释成已确认正常。

源码：[ml.cc L1195–1230](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1195)。

默认目标参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| training window | 6h | 每次查询最近的历史时间窗 |
| min training window | 15m | 根据采样间隔换算最低有效样本数 |
| train every | 3h | 训练队列目标循环周期 |
| number of models per dimension | 18 | 模型池容量 |
| max training vectors | 1440 | 随机采样的期望输出量，不是硬上限 |
| maximum number of k-means iterations | 1000 | 训练迭代上限 |

源码：[ml_config.cc L127–141](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_config.cc#L127)。

训练历史查询截至当前数据库最后一个点，最长取 6h；有效点数下限是 `max(ceil(15m / step), diff + smooth + lag)`。特征不足两条则不训练。采样比例按 `min(1, 1440 / expected_vectors)` 计算，在 lag 提取阶段按随机数筛选；因此 1440 是期望数量。

源码：[历史查询及点数 L215–310](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L215)、[采样与训练 L1027–1088](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1027)、[随机筛选](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_features.cc#L68)。

训练是后台队列循环：每个任务按 `train_every / (queue_size + 1)` 分配节流时间，单任务最多等 1 秒；任务执行后通常重新入队。实际间隔受队列大小、训练耗时、常量跳过等影响，不是严格每 3h 的定时网格。模型池未满时追加；满时通常轮换淘汰，但窗口重叠满足特定条件时会直接替换最新槽位。

源码：[训练队列 L1686–1733](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1686)、[模型池 L939–965](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L939)。

在“18 个满 6h 的训练窗，结束时间严格间隔 3h”的理想前提下，联合覆盖为 `6 + 17 * 3 = 57h`。这是窗口几何推导，不是源码保证的固定记忆长度或必须等待的冷启动时长。已有模型能从数据库加载，也会影响启动行为；[持久化查询](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L313)按 dimension 和时间载入最近的有限数量模型。

## 5. 缺失、常量与抑制

- 训练：缺失点用之前最后一个有效值前向填充，开头的 NaN 区段移除；真实有效点仍单独计数并要求达到下限。这不代表缺失值被建模为异常。[ml.cc L265–308](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L265)
- 预测：不存在的点清空最近原始样本缓存并返回 false，后续重新暖机。[ml.cc L1108–1114](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1108)
- 常量：训练预检查会跳过 `METRIC_TYPE_CONSTANT`；新值不同于最近值时标为 variable，训练周期完成后重设 constant。它是“自上次周期以来是否变化”的调度标记，不是额外的 MAD/std 检测器。[预检查](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L863)、[状态更新](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1142)
- 抑制：预测路径有异常累计/窗口累计及 `SILENCED` 状态，达到门槛后暂时不报告异常，后续训练会重置。不能只复制六步示意图便宣称逐点兼容 Netdata。[ml.cc L1189–1225](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml.cc#L1189)

## 6. 移植方案必须明确的边界

1. 以源码特征和距离公式为基准，但独立实现；Netdata 这些源文件声明 GPL-3.0-or-later，直接复制代码与仅参考算法是不同的工程决策。
2. 本项目若使用固定采样步长、严格的 3h 训练调度、确定性采样、最小模型数量门槛或禁用自动 silencing，应逐项记录为适配差异，而不是宣称与 Netdata 完全一致。
3. 原值上的上下阈值带无法由六维距离模型唯一反推；应以 anomaly score、anomaly bit、模型数量/训练状态替代旧的 baseline/bounds 语义。
4. 一位异常标记嵌入存储是 Netdata 时序引擎的实现能力，不是 k-means 的固有特性。本项目应先返回/保存明确的检测结果，再定义异常率的分母与缺失数据处理。
5. 不承诺 `10^-36` 误报率：min–max 分数不是校准概率，重叠训练窗和同一输入使模型相关；没有依据将 18 次投票概率独立相乘。应以本项目代表性指标集上的回放评估误报、漏报、检测延迟和资源占用。
6. 历史回放必须只使用评分时间之前完成训练的模型，避免用当前模型重算过去后误当成在线检测精度。

以上第 2–6 项为依据源码事实提出的本项目工程建议，不是 Netdata 官方行为保证。
