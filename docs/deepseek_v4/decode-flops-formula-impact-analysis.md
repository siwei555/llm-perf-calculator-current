# DeepSeek V4 Decode FLOPs 公式修正影响分析

## 1. 分析目的

本文比较 DeepSeek V4 当前 Decode FLOPs 工程近似与逐算子公式的差异，并评估公式修正对以下结果的影响：

- 单 token Decode FLOPs
- Decode 算力上限（Compute Ceiling）
- 最终 Decode TPS
- 权重、Cache、临时工作集及总显存

本分析只替换 `F_decode` 的计算口径，不修改模型参数、平台参数、权重流量、Cache 流量或显存公式。

## 2. 对比基线

### 2.1 工作负载

- Prompt/Decode 上下文：`131072 tokens`（128K）
- Batch Size：`1`

### 2.2 默认平台参数

| 参数 | 数值 |
|---|---:|
| Compute Throughput | 248 TFLOPS |
| Compute Efficiency | 40% |
| Effective Compute | 99.2 TFLOPS |
| Memory Bandwidth | 273 GB/s |
| Bandwidth Efficiency | 60% |
| Effective Bandwidth | 163.8 GB/s |
| Bytes / Weight | 1 byte |
| Bytes / Expert | 0.5 byte |
| Bytes / Activation | 2 bytes |
| Runtime Overhead | 4 GB |

### 2.3 统计边界

逐算子 Decode FLOPs 包含：

- Sparse MoE：Gate、Up、Down，以及每 token 激活的 routed experts 和 1 个 shared expert
- Q LoRA projection
- Compressed KV projection
- Sliding、CSA、HCA Attention core（QKᵀ + AV）
- CSA/HCA Compressor
- CSA Indexer projections 与 Indexer attention
- Grouped O projection

与计算器当前口径一致，以下小算子或输出路径暂不计入：

- RMSNorm/LayerNorm
- Softmax、激活函数和逐元素运算
- Router/Hash routing 的细粒度开销
- `lm_head` 全词表投影与 logits 后处理

## 3. 公式对比

### 3.1 原工程近似

```text
F_decode_old = L × D × I_moe × (K_active + 1) / 3
```

该公式只使用层数、隐藏维度、专家中间维度和激活专家数描述计算规模，没有逐项统计 MoE 的三个矩阵投影，也没有包含 Attention、Compressor 和 Indexer。

### 3.2 逐算子公式

```text
F_decode_new
  = N_sliding × F_sliding_decode
  + N_csa × F_csa_decode
  + N_hca × F_hca_decode
```

每类 Attention 层按实际算子求和：

```text
F_layer_decode
  = F_Q
  + F_KV
  + F_attention_core
  + F_compressor
  + F_indexer
  + F_O
  + F_MoE
```

其中标准 gated MoE 主体为：

```text
F_MoE = 6 × D × I_moe × (K_active + 1)
```

系数 `6` 来自 Gate、Up、Down 三个矩阵乘法，每个矩阵乘法按一次乘法和一次加法计 `2 FLOPs`。

## 4. 单 token Decode FLOPs 差异

| 模型 | 原估算 | 逐算子结果 | 新/旧倍数 | 增幅 |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 0.842 GFLOPs/token | 35.772 GFLOPs/token | 42.50× | 4150.1% |
| DeepSeek-V4-Pro | 3.134 GFLOPs/token | 121.815 GFLOPs/token | 38.87× | 3786.7% |

### 4.1 逐算子构成

| 模型 | MoE | Attention/Compressor/Indexer | 总计 |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 15.150 G | 20.622 G | 35.772 G |
| DeepSeek-V4-Pro | 56.415 G | 65.400 G | 121.815 G |

### 4.2 按层型构成

| 模型 | Sliding | CSA | HCA | 总计 |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 1.166 G | 20.093 G | 14.512 G | 35.772 G |
| DeepSeek-V4-Pro | 0 G | 64.735 G | 57.080 G | 121.815 G |

差异较大的主要原因：

- 原公式中的 `/3` 与标准 MoE 三投影 FLOPs 没有可追溯关系；仅 MoE 主体就比 `6 × D × I_moe × (K_active + 1)` 少算 18 倍。
- 原公式没有计算 Q/KV/O projection。
- 原公式没有计算 Sliding、CSA、HCA Attention core。
- 原公式没有计算 Compressor 和 CSA Indexer。
- 128K 下 CSA/HCA 的有效历史长度增大，使 Attention core 与 Indexer 的占比不可忽略。

## 5. 对算力上限的影响

算力上限公式为：

```text
TPS_compute = Effective Compute / F_decode
```

在 `248 TFLOPS × 40% = 99.2 TFLOPS` 有效算力下：

| 模型 | 原 Compute Ceiling | 修正后 Compute Ceiling | 降幅 |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 117862.74 token/s | 2773.15 token/s | 97.65% |
| DeepSeek-V4-Pro | 31650.88 token/s | 814.35 token/s | 97.43% |

公式修正会显著降低 Decode 的理论算力上限，使其不再被错误地显示为几万到十几万 token/s。

## 6. 对最终 Decode TPS 的影响

最终 Decode TPS 取算力上限和带宽上限的较小值：

```text
TPS_decode = min(TPS_compute, TPS_bandwidth)
```

128K 下的当前流量估算为：

| 模型 | Decode 权重流量 | Cache 流量 | 总流量/token | Bandwidth Ceiling |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 33.984 GB | 4.815 GB | 38.799 GB | 4.222 token/s |
| DeepSeek-V4-Pro | 110.938 GB | 18.421 GB | 129.359 GB | 1.266 token/s |

因此，在当前默认平台下：

| 模型 | 原最终 Decode TPS | 修正后最终 Decode TPS | 变化 |
|---|---:|---:|---:|
| DeepSeek-V4-Flash | 4.222 token/s | 4.222 token/s | 0 |
| DeepSeek-V4-Pro | 1.266 token/s | 1.266 token/s | 0 |

当前场景仍然是明显的带宽瓶颈。虽然 Compute Ceiling 大幅下降，但修正后的 Compute Ceiling 仍远高于 Bandwidth Ceiling，因此最终 TPS 暂时不变。

这不意味着 Decode FLOPs 可以继续使用旧近似。在更高带宽、更多设备并行、权重驻留分片或更高 Batch 的场景下，计算瓶颈可能变得可见，错误的 FLOPs 会直接导致 TPS 判断失真。

## 7. 对显存的影响

仅修正 Decode FLOPs 时，显存公式和所有显存结果均不变：

| 模型 | 权重显存 | 持久 Cache | 临时工作集 | Runtime Overhead | 总显存 | 修正前后差值 |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4-Flash | 115.000 GB | 0.908 GB | 0.151 GB | 4.000 GB | 120.059 GB | 0 GB |
| DeepSeek-V4-Pro | 800.000 GB | 1.299 GB | 0.302 GB | 4.000 GB | 805.601 GB | 0 GB |

原因如下：

- 权重显存由 checkpoint 参数量与 `Bytes / Weight`、`Bytes / Expert` 决定。
- 持久 Cache 由上下文长度、层数、压缩率和 Cache 精度决定。
- 单步临时工作集由 Attention 可见长度、head 数、head dimension 和 Activation 精度决定。
- `F_decode` 只用于计算 Compute Ceiling，不参与上述显存公式。

因此，本次公式修正对显存的直接影响为 **0 GB**。如果未来同时调整算子实现、缓存布局、激活精度或并行策略，则需要另行评估显存变化。

## 8. 结论

- 原 `/3` 近似严重低估 DeepSeek V4 Decode FLOPs。
- 128K 下，Flash 的逐算子结果约为原估算的 `42.50×`，Pro 约为 `38.87×`。
- 修正后 Compute Ceiling 分别下降到约 `2773.15 token/s` 和 `814.35 token/s`。
- 当前默认平台仍由内存带宽主导，因此最终 Decode TPS 暂时不变。
- Decode FLOPs 修正不改变权重、Cache、临时工作集或总显存，显存差值为 `0 GB`。
- 后续公式说明与计算引擎应统一使用逐算子结果，并删除无物理依据的 `/3` 工程近似。

