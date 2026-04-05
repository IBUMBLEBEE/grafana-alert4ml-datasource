# rsod-classifier 架构设计文档

## 1. 目标与动机

### 目标
设计一个时序数据类型检测器模块，能够自动识别输入时序数据的特征（平稳性、趋势、季节性等），并进行分类，为后续的异常检测、预测等任务提供数据驱动的算法选择依据。

### 动机
- **算法自适应**: 不同类型的时序数据需要不同的异常检测算法
  - 季节性数据应使用周期相关方法
  - 趋势数据应使用去趋势方法
  - 平稳数据应使用基线方法
- **可解释性**: 用户能理解系统为何选择特定算法
- **可维护性**: 基于 sklearn Pipeline 的分阶段设计便于拓展和测试

## 2. 架构设计

### 2.1 分层架构

```
┌─────────────────────────────────────────┐
│  应用层 (Application Layer)              │
│  - 异常检测选择                         │
│  - 特征工程选择                         │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Pipeline 层 (Pipeline Layer)            │
│  TimeSeriesClassifierPipeline            │
│  - 协调各阶段                           │
│  - 聚合结果                             │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  阶段层 (Stage Layer)                    │
│  ┌─────────────┬──────────┬──────────┐  │
│  │ Preprocess  │ Analyze  │ Detect   │  │
│  └─────────────┴──────────┴──────────┘  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  算法层 (Algorithm Layer)                │
│  - preprocessing                        │
│  - stationarity (ADF/KPSS)              │
│  - trend (Linear Regression)            │
│  - seasonality (STL/ACF/FFT)            │
└─────────────────────────────────────────┘
```

### 2.2 模块组织

```
rsod-classifier/
├── src/
│   ├── lib.rs                 # 主入口，重新导出 API
│   ├── types.rs               # 数据类型定义
│   ├── traits.rs              # Pipeline trait 定义
│   ├── preprocessing.rs       # 数据预处理
│   ├── stationarity.rs        # ADF/KPSS 检验
│   ├── trend.rs               # 趋势检测
│   ├── seasonality.rs         # 季节性和周期性
│   └── pipeline.rs            # Pipeline 实现
├── Cargo.toml
└── README.md
```

## 3. 核心设计决策

### 3.1 Pipeline 模式 vs 传统函数链

**选择**: Pipeline 模式 (灵感来自 sklearn)

**原因**:
- ✓ 顺序控制清晰
- ✓ 易于调试（可在每个阶段打点）
- ✓ 易于扩展新阶段
- ✓ 配置管理集中

**实现**:
```rust
pub struct TimeSeriesClassifierPipeline {
    config: ClassifierConfig,
    results: RefCell<Option<ClassificationResult>>,
}
```

### 3.2 平稳性检验: 简化 vs 完整

**选择**: 简化实现（但预留完整实现接口）

**原因**:
- ✓ 减少依赖复杂度
- ✓ 对于 100-10000 点数据足够准确
- ✓ 可快速集成 anofox-forecast 的完整实现

**实现**:
```rust
pub fn simple_adf_test(values: &[f64], lags: usize) -> Result<(f64, f64)>
pub fn simple_kpss_test(values: &[f64]) -> Result<(f64, f64)>
```

### 3.3 多周期检测策略

**选择**: STL + FFT/ACF 混合方法

**原因**:
- ✓ STL 对单周期效果好
- ✓ FFT 对多周期频率提取好
- ✓ ACF 对周期延迟识别好

**实现**:
```rust
pub fn detect_seasonality_stl(values: &[f64]) -> Result<SeasonalityAnalysis>
pub fn detect_periodicity_fft(values: &[f64]) -> Result<PeriodicityAnalysis>
pub fn compute_acf(values: &[f64], max_lag: usize) -> Vec<f64>
```

### 3.4 决策规则的层级设计

**选择**: 嵌套 if-else + 强度阈值

**原因**:
- ✓ 生成可解释的推理过程
- ✓ 支持细粒度的置信度调节

**逻辑路径**:
```
CV 异常? → 返回 Irregular
非平稳? {
  有趋势? {
    有季节性? → SeasonalWithTrend
    否则 → Trending
  }
  有季节性? → Seasonal  
  否则 → Irregular
}
平稳? {
  有季节性? → Seasonal
  有周期性? → Stationary (低置信)
  否则 → Stationary
}
```

## 4. 算法选择依据

| 功能 | 算法 | 来源库 | 理由 |
|------|------|-------|------|
| 平稳性检验 | ADF + KPSS 对比测试 | statrs | 两个测试互补，robust |
| 趋势分析 | 线性回归 + Mann-Kendall | ndarray + 手工实现 | 经典组合，计算简单 |
| 季节性 | STL 分解 + ACF 峰值 | stlrs | 效果好，稳定 |
| 周期性 | FFT + ACF | rustfft | 频域+时域双重识别 |
| 数据处理 | DataFrame | polars (可选) | 便于批处理 |

## 5. 扩展机制

### 5.1 新阶段集成

```rust
pub trait ClassificationStage: Send + Sync {
    fn name(&self) -> &str;
    fn detect(&self, input: &ClassifierInput) -> Result<()>;
}
```

### 5.2 新算法集成示例

```rust
// 添加 TBATS 检测
pub fn detect_seasonality_tbats(data: &[f64]) -> Result<SeasonalityAnalysis> {
    // 调用 anofox-forecast 的 TBATS
    todo!()
}

// Pipeline 中使用
let tbats_result = detect_seasonality_tbats(values)?;
```

## 6. 性能分析

### 时间复杂度

| 阶段 | 复杂度 | 瓶颈 | 备注 |
|------|-------|------|------|
| 数据预处理 | O(n) | 线性扫描 | 缓存友好 |
| 平稳性检验 | O(n) | 线性回归 | 支持并行 |
| 趋势检测 | O(n log n) | 排序 | 可调参数 |
| 季节性 STL | O(n log n) | FFT | 可采样优化 |
| 周期性 FFT | O(n log n) | FFT | 已优化 |
| **总计** | **O(n log n)** | FFT | 对 10K 点 < 100ms |

### 内存复杂度: O(n)

```
输入数据: 8n bytes (f64 × 2)
中间结果: 8n bytes (ACF, 去趋势等)
输出: 1K bytes (结果结构)
---
总计: ~16n bytes (n=数据点数)
```

## 7. 测试策略

### 7.1 测试覆盖

```
┌─────────────────────────────┐
│      单元测试                │
├─────────────────────────────┤
│ • 平稳性: 常数 vs 趋势      │
│ • 趋势: 上升 vs 下降        │
│ • 季节性: 合成周期序列      │
│ • ACF/PACF: 已知周期        │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│      集成测试                │
├─────────────────────────────┤
│ • 实际异常检测数据集        │
│ • Grafana 面板数据          │
│ • IoT 传感器数据            │
└──────────┬──────────────────┘
           │
┌──────────▼──────────────────┐
│      回归测试                │
├─────────────────────────────┤
│ • 分类结果稳定性            │
│ • 置信度热图                │
│ • 算法运行时间              │
└─────────────────────────────┘
```

### 7.2 测试数据集

```rust
// 合成数据集生成
fn create_seasonal_series(periods: usize, amplitude: f64) -> Vec<f64> {
    (0..100)
        .map(|i| {
            let base = (i as f64 * 0.1).sin();
            let seasonal = amplitude * (2.0 * 3.14159 * i as f64 / periods as f64).sin();
            base + seasonal
        })
        .collect()
}

fn create_trending_series(slope: f64) -> Vec<f64> {
    (0..100).map(|i| i as f64 * slope).collect()
}
```

## 8. 集成建议

### 与 rsod-outlier 集成

```rust
use rsod_classifier::classify;
use rsod_outlier::outlier;

pub fn intelligent_outlier_detection(
    data: TimeSeriesInput<'_>,
) -> Result<DetectionResult> {
    // 1. 分类
    let classification = classify(data.timestamps, data.values)?;
    
    // 2. 根据分类选择周期
    let periods = match classification.classification {
        SeriesCharacteristic::Seasonal { periods } => periods,
        SeriesCharacteristic::SeasonalWithTrend { periods, .. } => periods,
        _ => vec![],
    };
    
    // 3. 使用周期进行无效检测
    outlier(data, &periods, "uuid")
}
```

### 与 rsod-forecaster 集成

```rust
pub fn intelligent_forecasting(
    data: TimeSeriesInput<'_>,
) -> Result<DetectionResult> {
    let classification = classify(data.timestamps, data.values)?;
    
    // 针对不同类型数据选择预测模型
    match classification.classification {
        SeriesCharacteristic::Seasonal { periods } => {
            // 使用 HoltWinters 或 SARIMA
            forecast_seasonal(data, periods)
        }
        SeriesCharacteristic::Trending(_) => {
            // 使用 线性模型或 Theta 方法
            forecast_trending(data)
        }
        _ => {
            // 使用 Naive 或均值方法
            forecast_stationary(data)
        }
    }
}
```

## 9. 常见问题与解决方案

### Q1: 如何处理多重周期数据？

**A**: 使用多次 STL 分解在残差上运行：

```rust
// 在 auto_mstl 中实现
for iteration in 0..max_iterations {
    let mstl_result = decompose(data, &periods);
    data = mstl_result.residual;  // 继续在残差中查找周期
}
```

### Q2: 时序数据不足 30 点怎么办？

**A**: 可有多个选项：

```rust
let config = ClassifierConfig {
    min_data_length: 10,  // 降低阈值
    ..Default::default()
};
```

或者返回"数据不足"的特殊分类。

### Q3: 如何适配不同的业务场景？

**A**: 使用 `ClassifierConfig` 的阈值参数：

```rust
// 对非常敏感的检测
let config = ClassifierConfig {
    seasonality_strength_threshold: 0.05,
    trend_pvalue_threshold: 0.1,
    ..Default::default()
};
```

## 10. 负债与改进计划

### 技术负债
- [ ] 简化的 ADF/KPSS 实现（应迁移到 anofox-forecast）
- [ ] 缺少 ARIMA 阶数建议
- [ ] 缺少异常值处理

### 短期改进（3 个月）
- [ ] 集成 anofox-forecast 完整检验
- [ ] 添加 MSTL 分解
- [ ] 性能基准测试

### 中期改进（6 个月）
- [ ] Web UI 可视化
- [ ] 实时流处理支持
- [ ] 模型序列化和缓存

### 长期改进（12 个月）
- [ ] 深度学习分类器（LSTM-VAE）
- [ ] 多元时序分类
- [ ] 自动机器学习集成

## 参考文献

1. [Time Series Forecasting with STL](https://robjhyndman.com/papers/JSS5605.pdf)
2. [Kwiatkowski-Phillips-Schmidt-Shin Test](https://en.wikipedia.org/wiki/KPSS_test)
3. [Augmented Dickey-Fuller Test](https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test)
4. [Mann-Kendall Trend Test](https://en.wikipedia.org/wiki/Kendall_tau_distance)
5. [scikit-learn Pipeline Design](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
