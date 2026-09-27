# `rsod-kmeans` 引入 `linfa-clustering` 的成本评估

研究日期：2026-09-27。

## 结论摘要

`linfa-clustering` 可以替换本项目“给定六维训练向量，求两个质心”这一小段训练内核，但不能替换特征构造、18 个历史窗口、训练样本选择、Netdata 风格的双质心平均距离评分、全模型共识、模型快照和 Grafana 请求状态处理。当前这些边界分别位于 [`features.rs`](../rsod/crates/rsod-kmeans/src/features.rs#L12)、[`engine.rs`](../rsod/crates/rsod-kmeans/src/engine.rs#L119)、[`model.rs`](../rsod/crates/rsod-kmeans/src/model.rs#L82)、[`ensemble.rs`](../rsod/crates/rsod-kmeans/src/ensemble.rs#L19) 和 [`storage.rs`](../rsod/crates/rsod-kmeans/src/storage.rs#L13)。

**判断：代码接入成本中等，行为迁移成本中高，不是无语义变化的依赖替换。** 最大成本不是把 `Vec<[f64; 6]>` 转成 `ndarray::Array2`，而是以下三项：

1. Linfa 0.8.1 的批量 `fit` 实现的是 m-k-means：每轮把旧质心当作额外样本参与均值；本项目当前实现的是普通 Lloyd 更新，并在空簇时把质心重置为离另一个质心最远的训练向量。相同种子也不能保证质心或告警结果不变。[Linfa 更新步骤](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L450-L478)；[本项目更新步骤](../rsod/crates/rsod-kmeans/src/model.rs#L51)。
2. Linfa 的公开批量 `fit` 循环没有取消回调。本项目当前在每次 Lloyd 迭代边界检查请求取消；直接替换会把最细取消粒度退化为“一个窗口模型训练前/后”。[Linfa `fit` 循环](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L229-L266)；[本项目迭代内取消](../rsod/crates/rsod-kmeans/src/model.rs#L37)。
3. Linfa 的默认值是 `n_runs=10`、`max_n_iterations=300`、`tolerance=1e-4`；本项目是固定种子的一次初始化、最多 1000 轮、固定 `1e-12` 收敛阈值。若不逐项显式配置，CPU 成本和结果都会变化。[Linfa 参数默认值](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/hyperparams.rs#L45-L89)；[本项目默认值](../rsod/crates/rsod-kmeans/src/config.rs#L21)。

因此不建议直接替换并立即作为默认实现。建议先做一个不改变持久化 DTO 的双跑原型；只有在代表性历史回放、取消延迟和资源基准都通过预先定义的验收标准后，才以模型版本 `v2` 灰度切换。若“保留当前质心结果和迭代级取消”是硬约束，则应继续保留当前小型训练内核，Linfa 不适合此次替换。

## 研究范围与版本

- Linfa 以已发布的 `linfa-clustering 0.8.1` 为评估对象，对应官方标签提交 [`4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0`](https://github.com/rust-ml/linfa/commit/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0)，而不是会继续变化的 `master`。官方文档当前也将 0.8.1 标为 latest：[docs.rs](https://docs.rs/linfa-clustering/0.8.1/linfa_clustering/)。
- 版本边界很重要：当前 `master` 已出现尚未包含在 0.8.1 中的 Hamerly 算法选择；本报告没有把未发布实现的行为或性能混入结论。[Linfa `master` 源码](https://github.com/rust-ml/linfa/blob/master/algorithms/linfa-clustering/src/k_means/algorithm.rs#L27-L31)
- Netdata 对照仍固定为项目已有研究采用的提交 [`4464a9ed71c8a7351b540ce4ca207e1418faffea`](https://github.com/netdata/netdata/commit/4464a9ed71c8a7351b540ce4ca207e1418faffea)。Netdata 的调用是 `dlib::pick_initial_centers` 加 `dlib::find_clusters_using_kmeans`，只能证明调用方式，不能证明与 Linfa 的 m-k-means 等价：[Netdata `ml_kmeans.cc`](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_kmeans.cc#L46-L102)。
- 本报告只做读取、依赖解析探针和源码比较，没有修改 `Cargo.toml`/`Cargo.lock`，也没有生成没有实测依据的耗时、吞吐或二进制体积数字。

## 可替换与不可替换边界

| 模块 | Linfa 能否替换 | 结论与依据 |
| --- | --- | --- |
| K-means++ 初始化、分配、质心迭代 | 能，但语义不同 | `KMeans::params_with_rng`、`n_runs`、`max_n_iterations`、`tolerance`、`init_method` 足以完成训练；可通过 `centroids()` 取回结果。[公开 API](https://docs.rs/linfa-clustering/0.8.1/linfa_clustering/struct.KMeans.html) |
| `diff(1) → SMA(3) → lag(5)` 六维特征 | 不能 | 这是项目/Netdata 的领域特征，不属于通用聚类库。[当前实现](../rsod/crates/rsod-kmeans/src/features.rs#L55)；[Netdata 特征源码](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_features.cc#L29-L109) |
| 18 个重叠时间窗、覆盖率、确定性限流采样 | 不能 | 项目按对齐 anchor 逐窗筛选，最多训练 18 个模型，并把每窗样本限制到 1440。[当前配置](../rsod/crates/rsod-kmeans/src/config.rs#L21)；[编排](../rsod/crates/rsod-kmeans/src/engine.rs#L131) |
| 异常分数 | 不能 | 项目计算到两个质心的欧氏距离均值，再用训练期 min/max 映射到 0–100；Linfa 的 `predict` 返回最近簇编号，`transform` 返回到最近质心的平方距离，二者都不是该评分。[当前评分](../rsod/crates/rsod-kmeans/src/model.rs#L82)；[Linfa `transform`/`predict`](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L387-L425) |
| 多模型共识 | 不能 | 项目取全部模型分数的最小值，等价于阈值上的全体同意。[当前实现](../rsod/crates/rsod-kmeans/src/ensemble.rs#L19) |
| 请求取消、单飞缓存、持久化 | 不能 | 这些属于服务生命周期而非聚类算法。[训练/加载编排](../rsod/crates/rsod-kmeans/src/engine.rs#L186)；[快照存储](../rsod/crates/rsod-kmeans/src/storage.rs#L22) |

最小侵入式方案应只替换 `KmeansModel::train_with_cancel` 内的初始化/迭代部分，训练完成后仍把两个质心复制到现有 `KmeansModel`，并沿用现有评分、ensemble 和存储结构。这能把 Rust 类型与 Linfa 类型限制在 `model.rs` 内，避免将 `ndarray` 或 `KMeans` 泄漏到项目公共接口。

## 依赖与许可证成本

### 直接依赖

`linfa-clustering 0.8.1` 的官方清单声明 `MIT OR Apache-2.0`，默认 feature 为空；它直接依赖 `ndarray`（启用 `rayon`、`approx`）、`linfa`、`linfa-nn`、`linfa-linalg`、`ndarray-rand`、`ndarray-stats`、`space` 等。[官方 `Cargo.toml`](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/Cargo.toml)。

本仓库 workspace 已有 `ndarray 0.16.1`、`rand 0.8` 和 Rayon，因此它们不是全新的依赖家族，但 `rsod-kmeans` 当前只直接依赖 `rand`、Serde、inventory 和内部 crate；接入时至少需要给该 crate 增加 `linfa-clustering`、`linfa`（导入 `Fit` trait）和 `ndarray`。[workspace 依赖](../rsod/Cargo.toml#L26)；[`rsod-kmeans` 当前清单](../rsod/crates/rsod-kmeans/Cargo.toml#L8)。可继续把当前 `StdRng` 传给 `params_with_rng`，不必再增加直接 `rand_xoshiro` 依赖。

在 `/tmp` 用 0.8.1 精确版本做 `cargo tree`，再按 crate 名与当前 [`rsod/Cargo.lock`](../rsod/Cargo.lock) 比较，出现了 11 个当前 lock 中没有的 crate 名：

```text
kdtree, linfa, linfa-clustering, linfa-linalg, linfa-nn,
ndarray-rand, ndarray-stats, noisy_float, pgat, space, sprs
```

这是“未实际修改 workspace”条件下的解析探针，不是最终 lock 承诺；真正接入后应以 `cargo tree -p rsod-kmeans` 和 lock diff 为准。`linfa-clustering` 没有按 K-means/DBSCAN/GMM 分算法 feature，因此即使只调用 K-means，crate 级依赖仍会进入解析与构建范围。[0.8.1 feature 列表](https://docs.rs/crate/linfa-clustering/0.8.1/features)。

### 许可证

本项目根许可证是 Apache-2.0：[LICENSE](../LICENSE)。Linfa 主 crate 和 clustering 是 `MIT OR Apache-2.0`，可选择 Apache-2.0 路径；探针中新出现的依赖也都是宽松许可证：

| 依赖组 | 包声明的许可证 | 一手来源 |
| --- | --- | --- |
| `linfa`, `linfa-clustering`, `linfa-nn`, `kdtree`, `ndarray-rand`, `sprs` | `MIT OR Apache-2.0` | [linfa](https://crates.io/crates/linfa/0.8.1)、[linfa-clustering](https://crates.io/crates/linfa-clustering/0.8.1)、[linfa-nn](https://crates.io/crates/linfa-nn/0.8.1)、[kdtree](https://crates.io/crates/kdtree/0.7.0)、[ndarray-rand](https://crates.io/crates/ndarray-rand/0.15.0)、[sprs](https://crates.io/crates/sprs/0.11.2) |
| `linfa-linalg`, `ndarray-stats` | `MIT/Apache-2.0`（旧式 SPDX 写法） | [linfa-linalg](https://crates.io/crates/linfa-linalg/0.2.1)、[ndarray-stats](https://crates.io/crates/ndarray-stats/0.6.0) |
| `noisy_float` | `Apache-2.0` | [noisy_float](https://crates.io/crates/noisy_float/0.2.1) |
| `pgat`, `space` | `MIT` | [pgat](https://crates.io/crates/pgat/0.3.0)、[space](https://crates.io/crates/space/0.19.0) |

这没有发现 GPL/AGPL 传播风险，但发布流程仍应生成最终 lock 对应的第三方 notices/SBOM；上述表不能代替最终依赖锁定后的许可证审计。相比之下，直接复制 Netdata 对应源文件会涉及其 GPL-3.0-or-later 源码声明，和使用独立实现的宽松许可证 Linfa 是不同决策。[Netdata 文件头](https://github.com/netdata/netdata/blob/4464a9ed71c8a7351b540ce4ca207e1418faffea/src/ml/ml_kmeans.cc#L1-L6)。

## API 与算法语义差异

| 项目 | 当前实现 | Linfa 0.8.1 | 迁移影响 |
| --- | --- | --- | --- |
| 初始化 | `StdRng(seed)`；首点均匀抽样；第二点按平方欧氏距离加权 | 默认 K-means++；允许传自定义 RNG；同样用加权索引和到最近质心的距离 | 可以显式传当前 `StdRng`、`KMeansPlusPlus`、`n_runs(1)`，但随机抽样实现和退化 fallback 并非 API 契约上的逐位兼容。[当前初始化](../rsod/crates/rsod-kmeans/src/model.rs#L115)；[Linfa 初始化](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/init.rs#L71-L130) |
| 更新公式 | 非空簇取样本算术均值 | m-k-means 把旧质心也计入分子、计数初值为 1 | 即使从完全相同初始质心开始，中间和最终质心也可能不同。[Linfa 源码](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L454-L478) |
| 空簇 | 选取离另一个质心最远的训练向量 | 空簇没有观测时保持旧质心；这是 m-k-means 更新的结果 | 退化、重复点和强不均衡数据上差异最大，必须建立专门夹具。[当前空簇](../rsod/crates/rsod-kmeans/src/model.rs#L65) |
| 收敛 | 两个质心各自移动的欧氏距离之和 `<= 1e-12` | 整个质心矩阵的 `Distance::distance` `< tolerance`；且参数校验要求 tolerance `> 0` | 可配置 `1e-12` 近似，但范数、严格比较符不同，不能宣称逐位兼容。[Linfa 循环](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L249-L265)；[参数校验](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/hyperparams.rs#L113-L129) |
| 多次启动 | 1 次 | 默认 10 次，取最小 inertia；可设为 1 | 必须显式 `n_runs(1)` 才接近当前成本和随机调用序列。[Linfa fit](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L237-L275) |
| 评分 | 到两个中心距离的均值，再做 min/max 映射 | 最近中心预测/最近中心平方距离 | 不能复用 Linfa `predict` 或 `transform` 代替项目评分，必须保留现有代码。 |
| 错误 | 项目 `RsodError`，训练前检查至少 2 条向量 | `KMeansParamsError`/`KMeansError`；参数 guard 检查 cluster/run/tolerance/iteration | 需增加错误映射；还应在调用前保持 `vectors.len() >= 2`，避免把领域错误语义交给库改变。[Linfa 错误类型](https://docs.rs/linfa-clustering/0.8.1/linfa_clustering/enum.KMeansError.html) |

## 确定性

Linfa 提供 `params_with_rng`，并在 `fit` 开头克隆所给 RNG；K-means++ 使用该 RNG 完成加权抽样，所以“同一输入、同一 Linfa/Rand 版本、同一参数和种子”具备可复现设计基础。[RNG API 与克隆](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L185-L197)、[`fit` L229–247](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L229-L247)。

但这不等于与当前实现结果一致，也不应把跨依赖升级的逐位结果视作未声明的稳定 API。迁移时应：

1. 显式传 `StdRng::seed_from_u64(options.seed ^ window_end as u64)`，保持现有每窗口 seed 派生规则。[当前 seed 派生](../rsod/crates/rsod-kmeans/src/engine.rs#L169)
2. 显式设置 `n_runs(1)`、`max_n_iterations(options.max_iterations)`、`KMeansPlusPlus` 和选定 tolerance；禁止依赖库默认值。
3. 在 lock 中精确固定 Linfa/Rand/ndarray 版本，并保留“同输入重复训练完全相等”的测试。[当前确定性测试](../rsod/crates/rsod-kmeans/src/model.rs#L176)
4. 质心比较必须允许两个中心交换顺序；异常评分对两个中心取均值，本身不依赖中心顺序。

Linfa 使用 Rayon 并行计算每个观测的最近质心，但质心累计更新仍是顺序执行；官方源码明确说明了这一边界。[并行说明](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L69-L82)、[并行 assignment](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L534-L549)。仍应在 CI 的实际目标平台上验证重复运行，因为本报告没有把浮点逐位一致性外推到所有编译目标。

## 取消语义

当前路径有三层取消检查：ensemble 开始、每个窗口开始、每次 Lloyd 迭代开始；后台还把并行 K-means 查询限制为两个 blocking worker。[engine 取消点](../rsod/crates/rsod-kmeans/src/engine.rs#L131)；[迭代取消点](../rsod/crates/rsod-kmeans/src/model.rs#L51)；[worker 限制](../rsod/crates/rsod-backend/src/pipeline.rs#L43)。

Linfa 的批量 `Fit::fit` 是一个不接受 callback/token 的同步调用，循环体也没有公开 hook。由源码可推断，直接采用后仍可在每个窗口模型的 `fit` 前后检查取消，但无法在该 `fit` 的内部迭代边界停止。[`Fit` 签名与循环](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L219-L292)。

可选方案及代价：

- **接受每模型粒度取消**：实现最简单，但构成可观察的服务行为退化；必须测最大训练输入和 1000 轮上取消到 CPU 停止的延迟。
- **使用 `fit_with` 分批并在批间检查**：公开 API 支持逐批调用，但那是 Mini-Batch K-means，更新公式和收敛语义进一步变化，不能作为等价取消补丁。[Linfa `fit_with`](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L295-L384)。
- **fork/上游增加取消 hook**：能保留粒度，但引入维护分叉；而 Linfa 内部 assignment/update helper 不是公共 API，项目无法只靠 wrapper 插入检查。
- **保留当前训练循环**：取消语义零回归，也避免为一个很小的核心引入通用聚类依赖；若取消是硬门槛，这是推荐选择。

## Serde 与持久化

Linfa 的 `KMeans` 在 `serde` feature 开启时可序列化，feature 还会打开 `ndarray/serde` 和 `linfa-nn/serde`。[模型 derive](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L13-L21)；[feature 声明](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/Cargo.toml#L24-L38)。但本项目不需要持久化完整 Linfa 对象：线上预测只需要两个中心、训练距离 min/max 和窗口边界，现有 `KmeansModel` 已准确表达这些字段。[现有 DTO](../rsod/crates/rsod-kmeans/src/model.rs#L10)。

0.8.1 的可序列化模型字段是 `centroids`、`cluster_count`、`inertia` 和 `dist_fn`，不包含训练超参数、RNG 或输入数据；该 struct 也没有显式的持久化格式版本。[Linfa 模型字段](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L178-L183)。因此即使技术上可序列化，也没有一手来源保证其 JSON 跨 Linfa 版本兼容。

推荐：**不要开启 Linfa `serde` feature，不要把 Linfa 类型写入数据库**。训练完成后通过 `centroids()` 复制为 `[[f64; 6]; 2]`，继续序列化当前 DTO。这样：

- 数据库格式不绑定 Linfa 私有字段、距离类型和 ndarray 表示；
- 将来更换训练库时仍能读取旧模型；
- 不增加 Serde feature 相关编译面；
- 现有快照原子发布和 round-trip 测试可继续使用。[当前 JSON 快照](../rsod/crates/rsod-kmeans/src/storage.rs#L13)。

尽管 JSON 结构可保持相同，训练算法语义已经改变。当前 `MODEL_VERSION=1`，存储 key 也包含 `kmeans:v1`；若上线 Linfa 结果，应升级模型/存储版本，防止进程在同一 rollout 中把旧算法生成的 v1 质心误当成新算法缓存复用。[模型版本](../rsod/crates/rsod-kmeans/src/engine.rs#L15)；[版本校验](../rsod/crates/rsod-kmeans/src/engine.rs#L113)；[v1 key](../rsod/crates/rsod-kmeans/src/storage.rs#L5)。迁移可以选择“v2 只读写新 key，旧 v1 留作回滚”，无需原地改写历史 blob。

## 性能与构建成本

源码本身不能证明 Linfa 在本项目负载上更快或更慢：

- 当前每个窗口最多 1440 个六维向量，最多 18 个模型，当前内核的 assignment 和 update 都是顺序循环。[配置](../rsod/crates/rsod-kmeans/src/config.rs#L21)；[当前循环](../rsod/crates/rsod-kmeans/src/model.rs#L51)
- Linfa 把 assignment 并行化、update 保持单线程；官方只描述机制，没有给出与本项目这种 `k=2, d=6, n<=1440` 工作负载对应的结论。[Linfa 并行说明](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L69-L82)
- 接入需要把 `&[[f64; 6]]` 复制/展开为 `Array2<f64>`，Linfa 每次 `fit` 还分配 memberships、distances 和 centroid arrays。[Linfa `fit` 分配](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/algorithms/linfa-clustering/src/k_means/algorithm.rs#L233-L247)
- 后端已经允许两个 blocking K-means 任务并行；每个任务再进入 Rayon 全局线程池可能产生嵌套并发/CPU 争用。这是需要基准验证的风险，不是已证实的退化。[worker 限制](../rsod/crates/rsod-backend/src/pipeline.rs#L43)
- 依赖探针显示新增通用数学/聚类 crate，会增加冷构建与供应链表面积。

为量化边界，本报告另在 `/tmp` 的隔离副本中做了一个**不进入仓库的临时 adapter**：只用 Linfa 替换质心训练，再把质心复制回现有 `KmeansModel`。结果如下：

| 指标 | 当前实现 | Linfa 临时 adapter | 差异/说明 |
| --- | ---: | ---: | --- |
| `model.rs` adapter diff | — | 23 行新增、75 行删除 | 净减少 52 行；不包含 Cargo lock、v2 版本和 replay 基础设施 |
| 现有 `rsod-kmeans` 测试 | 23/23 | 23/23 | 只证明现有契约未立即破坏，不证明质心/告警与当前实现等价 |
| musl release backend | 46,832,032 B | 46,926,112 B | **+94,080 B（+0.201%）** |
| 360 向量，单模型平均训练 | 约 0.0153 ms | 约 0.189 ms | Linfa 约 12.4 倍；按 18 模型约 3.4 ms |
| 1440 向量，单模型平均训练 | 约 0.0974 ms | 约 0.702 ms | Linfa 约 7.2 倍；按 18 模型约 12.6 ms |

微基准使用六维、`k=2`、固定 seed、各 200 次训练，并把 Linfa 所需的 `Array2` 转换计入候选耗时。它是本机合成数据上的方向性结果，不是 Criterion 基准，也没有覆盖高基数多序列、两个并发请求、缺失数据或生产 CPU。因此可得出的结论是：**相对训练开销明显增加，但单序列绝对耗时仍小；高基数场景必须再测，不能仅凭上述毫秒数通过性能门槛。**

构建数据也只作量级参考：最小独立 Linfa release 项目冷构建约 12.8 秒、产物目录约 134 MB；完整 backend 临时接入构建约 39.4 秒、峰值 RSS 约 669 MiB。但这两组测试的缓存与重新编译集合不同，不能与当前构建直接相减得出“Linfa 增加多少秒”。最终应在 CI 同一 clean cache 条件下做 A/B 构建。

正式决策仍应在同一 release profile、同一机器、同一输入 corpus 下测量：单窗口与 18 窗口 wall time、CPU time、峰值 RSS、分配次数（若现有工具支持）、并发两个查询时的吞吐/尾延迟、取消到 worker 停止的延迟、clean build 时间和最终 backend 文件大小。必须同时测 `RAYON_NUM_THREADS=1` 与部署默认线程池，避免把小数据并行开销或开发机核数误推广到生产。

## 迁移工作量（相对成本）

不提供人日估算；以下是可验证的工作包和相对复杂度。

| 工作包 | 相对成本 | 完成条件 |
| --- | --- | --- |
| Cargo 接入与类型转换 | 低 | 添加/锁定依赖；`FeatureVector` 与 `Array2` 双向转换；错误映射；`cargo check` 通过 |
| 训练 adapter | 低到中 | 参数全部显式设置；仅导出两个中心；现有评分/ensemble API 不变 |
| 算法行为验证 | 高 | 覆盖普通、常量、重复点、空簇、近距离 tie、强不均衡、达到最大迭代等夹具；记录差异而不是只验证“不崩溃” |
| 真实历史回放 | 高 | 对代表性指标逐点比较 score、阈值翻转、模型成熟状态和告警段；严禁未来数据泄漏 |
| 取消策略 | 中到高 | 明确接受每模型粒度，或保留自研循环/维护 fork；取消测试能证明选择的语义 |
| 持久化版本 | 中 | v2 key/version、v1 回滚策略、重启加载和并发快照测试通过 |
| 性能与构建验证 | 中 | 上一节指标有基线/候选可复现实测；无先验“库一定更快”结论 |
| 许可证/供应链 | 低到中 | 以最终 lock 运行许可证与漏洞审计，更新 notices/SBOM（如发布流程要求） |

编译层面的改动面较窄，主要集中在 `model.rs` 和 Cargo 清单；若采用 v2，还涉及 `engine.rs`/`storage.rs`。但输出质心会影响每个窗口的 min/max 标定和最终全体共识，因此业务行为面覆盖整个检测链，不能以修改文件数量衡量风险。

## 主要风险

1. **告警翻转风险（高）**：m-k-means、空簇和收敛范数不同，会改变质心；99 阈值加 18 模型一致性可能放大某一个窗口的小差异。依据是训练差异与 [`min` 共识](../rsod/crates/rsod-kmeans/src/ensemble.rs#L24)，不是已测得翻转率。
2. **取消退化风险（高）**：批量 `fit` 内无 hook，取消后的 CPU 工作可能继续到该模型结束。
3. **历史模型混用风险（高）**：不升版本会加载旧算法的 v1 模型，掩盖 rollout 行为并使回放难以解释。
4. **并发资源风险（中）**：blocking worker 并发与 Rayon 并行叠加；小型六维数据未必能摊薄调度开销，必须实测。
5. **维护收益被高估（中）**：Linfa 只接管训练内核，Netdata 风格的绝大多数领域代码仍由项目维护；同时新增 11 个此前 lock 中没有的 crate 名。
6. **“与 Netdata 等价”的错误表述（中）**：Netdata 调 dlib；Linfa 明确实现 m-k-means。即使两者都叫 K-means，也没有源码依据宣称相同质心。
7. **依赖升级可重复性（中）**：固定 seed 不代表跨 Linfa/Rand/ndarray 版本逐位稳定；升级必须重新跑差分 corpus。

## 推荐方案

推荐采用“先验证，后决定”的窄接入路径，而不是立即替换：

- 建立内部 `CentroidTrainer` seam，当前 trainer 和 Linfa trainer 输出同一个轻量 `[FeatureVector; 2]`；不要改变对外 `KmeansModel`。
- Linfa 候选固定 `0.8.1`，传当前 `StdRng`，显式 `k=2`、K-means++、`n_runs(1)`、当前迭代上限和选定 tolerance。
- 保留现有特征、采样、评分、ensemble、缓存和持久化 DTO；不要使用 Linfa `predict`、`transform` 或序列化模型替代领域逻辑。
- 候选只用于离线/影子双跑，先收集差分和资源数据。
- 只有产品明确接受算法变化，且每模型粒度取消满足 SLA 时，才上线 Linfa trainer，并使用 `MODEL_VERSION=2`/`kmeans:v2` 隔离缓存。
- 如果验收要求是“保持当前输出”或“请求取消必须在每轮可中断”，停止迁移并保留自研内核；这不是实现技巧可以无成本消除的差异。

## 分阶段验证计划

### 阶段 0：先冻结验收标准

在看候选结果前定义：允许的 score 偏差、告警位翻转容忍度、最大取消延迟、CPU/RSS/尾延迟预算、二进制和 clean-build 预算。数值应由产品和生产容量约束给出，本报告不臆造。

### 阶段 1：最小原型，不改线上格式

在独立 adapter 中接入 Linfa，只返回质心；保留现有 `KmeansModel` 和 JSON。增加参数快照测试，证明没有使用 Linfa 默认的 10 runs/300 iterations。此阶段不改默认 trainer。

### 阶段 2：合成差分测试

固定输入顺序和种子，双跑当前/Linfa trainer。至少覆盖：明显两簇、全常量、重复点、只有一个有效几何点、会产生空簇的数据、等距 tie、极端尺度、非有限值拒绝、最大迭代和多 seed。质心比较先做二中心排列匹配，再比较中心、训练 min/max、逐点 score 和 threshold decision。

### 阶段 3：真实历史 replay

用代表性指标和真实缺失模式重放完整 18 窗口流程。每个评分时刻只能使用当时已经训练完成的历史窗口。输出按指标类别分层的 score 差异、告警位翻转、连续告警段变化、暖机/NoData 变化；不能只报告平均值。

### 阶段 4：运行与供应链验证

在 release 构建测量单查询、两个并发查询、取消中的最坏输入和 Rayon 线程数敏感性；比较最终 backend 大小和 clean build。对最终 `Cargo.lock` 运行依赖许可证/漏洞审计，并检查目标平台/MSRV CI。Linfa 0.8.0 将 MSRV 提升到 Rust 1.87；当前开发环境虽高于此版本，仓库没有声明统一 `rust-version`，仍需目标构建验证。[Linfa changelog](https://github.com/rust-ml/linfa/blob/4484a55c8eb3105fcab5f6d5ed0c4cc0d2c2bbb0/CHANGELOG.md#version-080---2025-09-30)。

### 阶段 5：v2 灰度与回滚

使用独立 v2 key/version，按少量指标或实例灰度；保留 v1 读取/回滚能力但不混用两代模型；观测资源与告警差异达到阶段 0 标准后再扩大。若取消、性能或告警行为任一项超标，回退 trainer 和模型版本，不做数据库原地转换。

## 最终决策条件

| 条件 | 决策 |
| --- | --- |
| 目标是减少自研算法代码，允许 m-k-means 行为变化，并接受每模型粒度取消 | 完成上述双跑与 v2 计划后可以引入 |
| 要求当前质心/告警结果兼容 | 不引入；Linfa 0.8.1 不是 drop-in replacement |
| 要求迭代级协作取消 | 不直接采用批量 `fit`；保留当前内核，或先推动可维护的上游取消 API |
| 只因“成熟库应该更快/更稳定”而迁移，但没有 replay 与 benchmark | 不应引入；现有一手源码不能支持该结论 |
