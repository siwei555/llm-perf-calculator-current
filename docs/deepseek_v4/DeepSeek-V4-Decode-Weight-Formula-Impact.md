# DeepSeek V4 Decode 权重流量公式变更影响分析

## 1. 结论摘要

DeepSeek V4 的 Decode 权重流量由旧说明：

```text
B_weights ≈ M_weights
```

修正为：

```text
B_weights
= N_non × 10^9 × bpw
+ N_exp × 10^9 × (k / E) × bpe
```

本次修正解决了“每 token 权重读取流量”和“全部权重常驻显存”混用的问题。

- DeepSeek-V4-Flash 的每 token 权重流量从 `115.000 GB` 降至
  `33.984 GB`，减少 `81.016 GB`，降幅 `70.45%`。
- DeepSeek-V4-Pro 的每 token 权重流量从 `800.000 GB` 降至
  `110.938 GB`，减少 `689.063 GB`，降幅 `86.13%`。
- Decode FLOPs、算力上限、Prefill 计算量、持久 cache 和临时工作集不因
  该权重流量公式本身发生变化。
- 模型全部权重仍必须常驻设备或分布式设备组；新公式不能用于降低常驻权重显存估算。

## 2. 对比范围与假设

模型参数取自当前模型注册表：

- `src/engines/model-registry/deepseekV4Models.ts`

使用模型推荐精度：

| 参数 | 数值 | 含义 |
|---|---:|---|
| `bpw` | 1 byte | 普通权重采用 FP8 |
| `bpe` | 0.5 byte | 专家权重采用 FP4 |
| `Bytes / Activation` | 2 bytes | Activation/cache 采用 BF16 |

权重数值按计算引擎当前的动态精度公式计算，即 `N_non × bpw + N_exp × bpe`；
模型定义中的静态 `estimatedWeightsGb` 不参与本次对比。特别是 Flash 的静态估计值
`145.82 GB` 与上述动态精度结果 `115 GB` 口径不同，不能混入新旧公式差值。

带宽示例使用工具默认平台参数：

```text
Memory Bandwidth = 273 GB/s
Bandwidth Efficiency = 60%
Effective Bandwidth = 163.8 GB/s
Batch Size = 1
Context = 131072 tokens（128K）
```

本文中的 GB 使用工具当前采用的十进制单位：

```text
1 GB = 10^9 bytes
```

## 3. 两个公式分别代表什么

### 3.1 常驻权重显存

```text
M_weights
= N_non × 10^9 × bpw
+ N_exp × 10^9 × bpe
```

`M_weights` 包含所有非专家权重和所有专家权重。即使一个 token 只激活少量专家，
未激活专家通常仍需要驻留在本卡、其他卡或分布式存储层级中。

### 3.2 每 token 权重读取流量

```text
B_weights
= N_non × 10^9 × bpw
+ N_exp × 10^9 × (k / E) × bpe
```

每个 token 会读取全部非专家路径，但只读取被路由选中的 `k` 个专家。因此专家权重
流量按 `k/E` 计入，而不是按全部专家计入。

两者的差值为：

```text
M_weights - B_weights
= N_exp × 10^9 × (1 - k/E) × bpe
```

该差值是当前 token 未访问的专家权重，不是可以从常驻显存中直接删除的权重。

## 4. 模型参数与计算结果

| 模型 | `N_non` | `N_exp` | `k / E` | 旧公式 `B_weights` | 新公式 `B_weights` | 减少量 | 降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 30B | 170B | 6/128 = 4.6875% | 115.000 GB | 33.984 GB | 81.016 GB | 70.45% |
| DeepSeek-V4-Pro | 100B | 1400B | 6/384 = 1.5625% | 800.000 GB | 110.938 GB | 689.063 GB | 86.13% |

### 4.1 DeepSeek-V4-Flash

旧公式：

```text
B_weights_old
= 30 × 1 + 170 × 0.5
= 115.000 GB/token
```

新公式：

```text
B_weights_new
= 30 × 1 + 170 × (6/128) × 0.5
= 33.984375 GB/token
```

新旧比值：

```text
B_weights_old / B_weights_new = 3.384
```

旧公式把每 token 权重流量高估约 `3.38` 倍。

### 4.2 DeepSeek-V4-Pro

旧公式：

```text
B_weights_old
= 100 × 1 + 1400 × 0.5
= 800.000 GB/token
```

新公式：

```text
B_weights_new
= 100 × 1 + 1400 × (6/384) × 0.5
= 110.9375 GB/token
```

新旧比值：

```text
B_weights_old / B_weights_new = 7.211
```

旧公式把每 token 权重流量高估约 `7.21` 倍。

## 5. 对 Decode TPS 的影响

Decode 带宽上限为：

```text
TPS_decode_bandwidth
= Effective Bandwidth / (B_weights + B_cache)
```

因此新公式只改变分母中的 `B_weights`。在 128K、Batch=1、BF16 cache 和默认有效
带宽 `163.8 GB/s` 下，当前引擎的 Decode cache 流量及带宽上限为：

| 模型 | 128K Decode cache 流量/token | 旧带宽上限 | 新带宽上限 | 提升倍数 |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 4.815 GB | 1.367 token/s | 4.222 token/s | 3.09× |
| DeepSeek-V4-Pro | 18.421 GB | 0.200 token/s | 1.266 token/s | 6.33× |

这里的提升是“带宽上限”的变化，不保证最终 Decode TPS 一定按相同比例提升：

```text
TPS_decode
= min(TPS_decode_compute, TPS_decode_bandwidth)
```

如果修正后仍是带宽瓶颈，最终 TPS 会接近新的带宽上限；如果瓶颈切换为算力，最终
TPS 将被 `TPS_decode_compute` 截断。

## 6. 对算力估算的影响

该公式描述的是权重字节流量，不是 FLOPs。因此它不会直接改变：

- Sliding、CSA、HCA Attention FLOPs；
- Compressor FLOPs；
- Indexer linear/attention FLOPs；
- MoE 激活专家 FLOPs；
- Decode compute ceiling；
- Prefill FLOPs 和 Prefill compute ceiling。

DeepSeek V4 Decode FLOPs 从整体近似改为逐算子公式是另一项独立变更，不应把由此产生
的算力侧变化归因于 `B_weights` 公式修正。

## 7. 对权重与显存估算的影响

### 7.1 不变的项目

正确实现下，以下项目不应因新公式而变化：

- `M_weights`：全部模型权重常驻显存；
- Persistent Sliding/CSA/HCA cache；
- Single-step temporary working set；
- Runtime overhead；
- Decode peak runtime memory。

按当前动态精度公式，常驻权重仍为：

| 模型 | 常驻普通权重 | 常驻专家权重 | `M_weights` |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 30 GB | 85 GB | 115 GB |
| DeepSeek-V4-Pro | 100 GB | 700 GB | 800 GB |

### 7.2 128K 默认场景的总显存

在 Batch=1、BF16 cache、Runtime Overhead=4 GB 下：

| 模型 | 常驻权重 | Persistent cache | Temp peak | Runtime overhead | 总显存估算 |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 115.000 GB | 0.908 GB | 0.151 GB | 4.000 GB | 120.059 GB |
| DeepSeek-V4-Pro | 800.000 GB | 1.299 GB | 0.302 GB | 4.000 GB | 805.601 GB |

这些总显存数值在新旧 Decode 权重流量公式之间保持不变。

### 7.3 必须避免的错误用法

如果错误地把新的 `B_weights` 当成 `M_weights`，会造成：

- Flash 常驻权重少算 `81.016 GB`；
- Pro 常驻权重少算 `689.063 GB`；
- 错误判断单卡或设备组是否能够容纳模型；
- 忽略未激活专家仍需驻留或由其他设备承载的事实。

因此界面和报告中应始终使用两个不同符号：

```text
M_weights              # 常驻权重显存
B_decode_weight_read   # 每 token Decode 权重读取流量
```

当前代码沿用 `B_weights` 表示第二项，但其含义必须在公式变量表中明确。

## 8. 对其他结果项的影响汇总

| 结果项 | 是否受影响 | 影响 |
|---|---|---|
| Decode 权重流量 | 是 | Flash 降低 70.45%，Pro 降低 86.13% |
| Decode bandwidth ceiling | 是 | 128K 默认示例分别提高约 3.09×、6.33× |
| 最终 Decode TPS | 可能 | 仅在带宽仍为瓶颈时接近上述提升 |
| Decode FLOPs | 否 | 与权重字节流量公式无关 |
| Decode compute ceiling | 否 | 由有效算力和 FLOPs 决定 |
| 常驻权重显存 | 否 | 仍包含全部专家权重 |
| Persistent cache | 否 | 由上下文长度、层类型和 cache 精度决定 |
| 临时工作集 | 否 | 由 Attention 可见长度和 activation 精度决定 |
| 总运行显存 | 否 | 不应使用每 token 权重流量替代常驻权重 |
| Prefill TPS/FLOPs | 否 | 本次公式只属于 Decode 权重流量 |

## 9. 口径限制

active-expert 公式是理想化的每 token 平均权重读取模型，还没有额外计入：

- expert routing 不均衡；
- 同一 batch 内不同 token 激活专家集合的并集扩大；
- tensor/expert parallel 通信；
- 权重缓存命中、预取和片上 SRAM 复用；
- 量化 scale、zero point、元数据和未量化模块；
- 跨卡专家访问产生的链路流量。

因此新公式比 `B_weights ≈ M_weights` 更符合 MoE 的算子访问逻辑，但仍属于理论性能
估算，不等同于具体推理框架的实测 HBM/统一内存流量。
