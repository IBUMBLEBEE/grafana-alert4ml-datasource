# rsod-classifier: 时序数据类型检测器

## 总览

`rsod-classifier` 是一个用于自动检测和分类时序数据类型的 Rust 库。它基于多阶段 Pipeline（类似 sklearn Pipeline），通过统计分析、平稳性检验、趋势检测和季节性分析来识别时序数据的特征。

## 支持的分类类型

- **Stationary（平稳）**: 无明显趋势、季节性或周期性的数据
- **Trending（趋势）**: 具有明显上升或下降趋势的数据
- **Seasonal（季节性）**: 存在规律性季节模式的数据
- **SeasonalWithTrend（季节+趋势）**: 同时具有季节性和趋势的数据
- **Irregular/Noisy（不规则/噪声）**: 高方差、无规律的数据

## Pipeline 架构

分类流程分为 7 个阶段：

```
输入时序数据
    ↓
[1] 数据预处理和验证
    - 检查缺失值率
    - 处理异常值
    ↓
[2] 基础统计特征提取
    - 均值、方差、偏度、峰度
    - 变异系数 (CV)
    ↓
[3] 平稳性检验
    - ADF (Augmented Dickey-Fuller) 测试
    - KPSS (Kwiatkowski-Phillips-Schmidt-Shin) 测试
    ↓
[4] 趋势检测
    - 线性回归斜率分析
    - Mann-Kendall 检验
    ↓
[5] 季节性检测
    - STL 分解
    - 季节强度计算
    ↓
[6] 周期性检测
    - FFT 频谱分析
    - 自相关函数 (ACF)
    ↓
[7] 综合分类和决策
    - 应用决策规则
    - 输出分类结果和置信度
    ↓
输出: SeriesCharacteristic 及详细分析结果
```

## 快速开始

### 基础使用

```rust
use rsod_classifier::classify;

fn main() {
    // 准备数据
    let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let values: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 10.0).collect();

    // 分类
    let result = classify(&timestamps, &values).unwrap();

    println!("分类: {:?}", result.classification);
    println!("置信度: {:.2}", result.confidence);
    println!("推理: {}", result.reasoning);
    println!("详细结果:");
    println!("  - CV: {:.4}", result.coefficient_of_variation);
    if let Some(stationarity) = &result.stationarity {
        println!("  - 平稳性 (ADF p-value): {:.4}", stationarity.adf_pvalue);
    }
    if let Some(trend) = &result.trend {
        println!("  - 趋势斜率: {:.6}", trend.slope);
    }
    if let Some(seasonality) = &result.seasonality {
        println!("  - 季节强度: {:.4}", seasonality.strength);
        println!("  - 检测周期: {:?}", seasonality.periods);
    }
}
```

### 使用自定义配置

```rust
use rsod_classifier::{classify_with_config, ClassifierConfig};

fn main() {
    let timestamps: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let values: Vec<f64> = vec![5.0; 50];

    // 创建自定义配置
    let mut config = ClassifierConfig::default();
    config.seasonality_strength_threshold = 0.2;  // 提高季节性检测阈值
    config.max_seasonal_period = 24;  // 最大检测 24 小时周期
    config.use_fft = true;  // 启用 FFT 分析
    
    let result = classify_with_config(&timestamps, &values, config).unwrap();
    println!("分类: {:?}", result.classification);
}
```

### 管道方式（高级）

```rust
use rsod_classifier::{TimeSeriesClassifierPipeline, ClassifierInput};

fn main() {
    let timestamps: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let values: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();

    let classifier = TimeSeriesClassifierPipeline::new();
    let input = ClassifierInput::new(&timestamps, &values);
    
    let result = classifier.classify(&input).unwrap();
    println!("分类结果: {:#?}", result);

    // 获取最后的结果
    if let Some(last) = classifier.last_result() {
        println!("缓存结果: {:#?}", last);
    }
}
```

## 配置选项

`ClassifierConfig` 提供以下配置参数：

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|------|------|
| `stationarity_method` | String | "both" | 平稳性检验方法: "adf", "kpss", "both" |
| `adf_significance` | f64 | 0.05 | ADF 检验显著性水平 |
| `kpss_significance` | f64 | 0.05 | KPSS 检验显著性水平 |
| `trend_pvalue_threshold` | f64 | 0.05 | 趋势显著性 p 值阈值 |
| `seasonality_strength_threshold` | f64 | 0.1 | 季节性强度阈值 (0-1) |
| `irregular_cv_threshold` | f64 | 0.8 | 不规则数据变异系数阈值 |
| `max_seasonal_period` | usize | 336 | 最大检测周期 |
| `min_data_length` | usize | 30 | 最小数据点数 |
| `use_fft` | bool | true | 是否使用 FFT 周期性检测 |
| `use_acf` | bool | false | 是否使用 ACF 分析 |

## 输出说明

`ClassificationResult` 包含完整的分析结果：

```rust
pub struct ClassificationResult {
    pub data_stats: DataStatistics,           // 基础统计
    pub stationarity: Option<StationarityTest>, // 平稳性测试结果
    pub trend: Option<TrendAnalysis>,         // 趋势分析
    pub seasonality: Option<SeasonalityAnalysis>, // 季节性分析
    pub periodicity: Option<PeriodicityAnalysis>, // 周期性分析
    pub coefficient_of_variation: f64,        // 变异系数
    pub classification: SeriesCharacteristic,  // 最终分类
    pub confidence: f64,                       // 置信度 (0-1)
    pub reasoning: String,                    // 分类理由
}
```

## 工作原理

### 平稳性检验

- **ADF Test**: 检验原假设（非平稳）。p value < 0.05 表示序列为平稳。
- **KPSS Test**: 检验原假设（平稳）。p value > 0.05 表示序列为平稳。

### 趋势检测

- 使用线性回归计算斜率，判断上升/下降趋势
- 关键指标：t-统计量、p-value、趋势强度
- 通过 Mann-Kendall 检验验证趋势显著性

### 季节性检测

- STL 分解识别季节分量
- 计算季节强度 = 1 - Var(残差) / Var(季节+残差)
- 检验常见周期：7天、24小时等

### 周期性检测

- FFT 频域分析识别主导频率
- ACF 自相关分析识别周期
- 找到显著的峰值对应的周期

### 决策规则

```
IF CV > 0.8:
    分类为 Irregular (置信度: 0.3)
ELSE IF 非平稳:
    IF 有趋势:
        IF 有季节性:
            分类为 SeasonalWithTrend (置信度: 0.85)
        ELSE:
            分类为 Trending (置信度: 0.80)
    ELSE IF 有季节性:
        分类为 Seasonal (置信度: 0.75)
    ELSE:
        分类为 Stationary (置信度: 0.50)
ELSE:  # 平稳
    IF 有季节性:
        分类为 Seasonal (置信度: 0.85)
    ELSE IF 有周期性:
        分类为 Stationary (置信度: 0.70)
    ELSE:
        分类为 Stationary (置信度: 0.90)
```

## 集成与应用

### 与异常检测的结合

```rust
use rsod_classifier::classify;
use rsod_outlier::outlier;

fn anomaly_detection_with_classification(
    timestamps: &[f64],
    values: &[f64],
    periods: &[usize],
) -> Result<()> {
    // 第一步：分类时序
    let classification = classify(timestamps, values)?;
    println!("时序类型: {:?}", classification.classification);

    // 第二步：根据分类选择异常检测方法
    match classification.classification {
        SeriesCharacteristic::Seasonal { ref periods } => {
            // 对于季节性数据，使用周期相关的异常检测
            let result = outlier(
                TimeSeriesInput::new(timestamps, values),
                periods,
                "model-uuid",
            )?;
            println!("异常得分: {:?}", result.anomalies);
        }
        SeriesCharacteristic::Trending(_) => {
            // 对于趋势数据，使用去趋势的异常检测
            println!("使用去趋势方法进行异常检测");
        }
        _ => {
            // 其他情况使用默认方法
            println!("使用默认异常检测方法");
        }
    }

    Ok(())
}
```

## 性能考虑

- **数据长度**: 建议 100-10000 点
- **计算复杂度**: O(n log n) for FFT, O(n²) for ACF
- **内存使用**: 约 8n 字节（n = 数据点数）

## 局限与改进方向

### 当前限制

1. 简化的 ADF/KPSS 实现（生产使用建议使用 anofox-forecast）
2. 对多重周期的检测能力有限
3. 不支持多变量时序分类

### 未来改进

- [ ] 集成 anofox-forecast 的完整 ADF/KPSS 实现
- [ ] 支持多周期检测和 MSTL 分解
- [ ] 添加自定义特征提取插件系统
- [ ] Web UI 用于交互式分类和可视化
- [ ] 模型序列化和跨会话缓存

## 依赖项

- `rsod-core`: 核心类型和 traits
- `statrs`: 统计函数和分布
- `stlrs`: STL 分解
- `augurs`: 时序分析工具
- `ndarray`: 数值计算

## 测试

运行测试：

```bash
cargo test -p rsod-classifier
```

覆盖的场景：

- ✓ 常数序列（平稳）  
- ✓ 趋势序列（上升/下降）
- ✓ 周期序列
- ✓ 季节性检测
- ✓ 缺失值处理
- ✓ 统计计算
- ✓ ACF/PACF 计算

## 贡献

欢迎提交 Bug 报告和功能建议！

## 许可证

与 rsod 项目相同
