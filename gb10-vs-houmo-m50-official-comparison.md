# NVIDIA GB10 与后摩漫界 M50 官方规格对比分析

> 更新日期：2026-08-21  
> 数据原则：仅使用 NVIDIA、后摩智能官网及 NVIDIA 官方技术文档公开的信息。未公开项目明确标为“官方未披露”，不使用第三方估算或由其他型号反推。

## 1. 对比对象

- **NVIDIA GB10**：Grace Blackwell Superchip，本文采用 NVIDIA DGX Spark 官方规格。GB10 是包含 Blackwell GPU、20 核 Arm CPU 与统一内存子系统的 SoC。
- **后摩漫界 M50**：后摩智能面向智能计算场景的存算一体芯片。

这里的 M50 指“后摩漫界 M50”，不是海光 DCU，也不是其他厂商的同名产品。

## 2. 官方规格对比

| 项目 | NVIDIA GB10（DGX Spark） | 后摩漫界 M50 | 可比性说明 |
| --- | --- | --- | --- |
| 产品形态 | Grace Blackwell SoC，集成 GPU 与 20 核 Arm CPU | 存算一体 AI 芯片 | 产品定位与系统边界不同 |
| 官方 AI 算力 | 最高 1 PFLOP FP4，使用稀疏特性 | 160 TOPS INT8；100 TFLOPS bFP16 | 精度与稀疏口径不同，不能直接相除 |
| 内存容量 | 128 GB LPDDR5X 一致性统一系统内存 | 最高 48 GB LPDDR5 | GB10 为 CPU/GPU 共享内存，不等同于独立显存 |
| 内存位宽 | 256-bit | 最高 192-bit | 可按官方标称值比较 |
| 内存带宽 | 273 GB/s | 最高 153.6 GB/s | GB10 约为 M50 的 1.78 倍，仅为标称带宽算术比值 |
| 功耗 | GB10 SoC TDP 140 W | 典型功耗 10 W | TDP 与典型功耗不是同一测试口径 |
| CUDA Core | 6,144 | 不适用/官网未按 CUDA Core 披露 | 架构不同 |
| Tensor Core | 第五代 Tensor Core | 官网未按 Tensor Core 口径披露 | 架构不同 |
| 主机通道 | SoC 集成；官网未给出离散加速卡式主机通道 | PCIe Gen4 ×4 | 不能视为同形态 PCIe 卡对比 |
| 片间/设备互联 | DGX Spark 配备 ConnectX-7，200 Gbps | HM-Link，16 GB/s | 一个是系统网络接口，一个是芯片互联，定义不同 |
| 官方列出的 M50 数据类型 | — | INT8、INT16、FP16、FP32、bFP16、bFP24 | 后摩官网明确披露 |

## 3. GB10 是否只支持 FP4

不是。**FP4 只是 NVIDIA DGX Spark 产品规格页公开峰值算力的精度，并不表示 GB10 只支持 FP4。**

NVIDIA 官方资料能够确认：

- GB10 属于 CUDA Compute Capability 12.1，包含 6,144 个 CUDA Core 和第五代 Tensor Core。
- NVIDIA 官方 NIM 支持矩阵存在 GB10 的 BF16 配置。
- NVIDIA 官方 Visual GenAI NIM 列出了 DGX Spark 的 FP8 配置。
- NVIDIA Blackwell/CUTLASS 软件栈支持 FP64、FP32、TF32、FP16、BF16、FP8、NVFP4、MXFP4/MXFP6/MXFP8 以及多种整数格式。

但“软件与硬件支持某种精度”和“厂商公开该精度的峰值算力”是两回事。当前官方公开资料中，GB10 的明确 SKU 峰值是：

```text
最高 1 PFLOP FP4（使用稀疏特性）
```

GB10 的 FP8、BF16、FP16、TF32、FP32、INT8 等峰值，DGX Spark 公开规格表未给出。因此不能把 1 PFLOP 简单除以 2、4 或其他系数来生成这些数值。

## 4. 算力对比为何不能直接给出胜负倍率

### 4.1 官方数字使用不同精度

```text
GB10：1 PFLOP FP4，带稀疏特性
M50：100 TFLOPS bFP16
M50：160 TOPS INT8
```

FP4、bFP16 与 INT8 的位宽、动态范围、运算指令、累加格式和适用负载不同。GB10 的数字还包含稀疏加速条件。因此以下说法均不成立：

- “GB10 算力是 M50 的 10 倍”；
- “GB10 FP8 必然是 500 TFLOPS”；
- “GB10 BF16 必然是 250 TFLOPS”；
- 根据其他 Blackwell GPU 的精度比例反推 GB10。

### 4.2 可以怎样做严谨比较

若要比较实际大模型推理性能，应在相同条件下测量：

- 相同模型与权重精度；
- 相同 Prompt、Decode 长度和 Batch；
- 相同稀疏/非稀疏设置；
- 相同算子覆盖范围；
- 相同 TPS、TTFT 和功耗测量方法；
- 明确是否包含 CPU、数据搬运和框架开销。

在缺少上述同口径实测数据时，只能列出官方规格，不能生成可信的算力倍率。

## 5. 内存与显存分析

### 5.1 容量

- GB10：128 GB LPDDR5X 一致性统一系统内存。
- M50：最高 48 GB LPDDR5。

按标称容量计算，GB10 是 M50 最大配置的约 2.67 倍。但必须注意：GB10 的 128 GB 由 CPU 与 GPU 共享，系统与显示保留也会占用其中一部分；它不能直接当作传统离散 GPU 的 128 GB 专用显存。

### 5.2 带宽

- GB10：273 GB/s。
- M50：最高 153.6 GB/s。

标称带宽算术比值：

```text
273 / 153.6 ≈ 1.78
```

该比值只比较内存接口的理论标称带宽，不等于模型推理 TPS 比值。实际性能还受模型精度、算子实现、计算吞吐、缓存、调度与软件栈影响。

## 6. 功耗与能效分析

- GB10 官网给出的是 **SoC TDP 140 W**，覆盖 CPU 与 GPU。
- M50 官网给出的是 **典型功耗 10 W**。

二者不是同一种功耗定义，且公开算力也不是同精度，因此不能使用 `官方算力 ÷ 官方功耗` 得出可信的跨芯片能效结论。

可以做出的定性判断是：M50 官方定位明显偏向低功耗边缘/嵌入式智能计算；GB10 面向桌面大模型开发、推理与微调，并提供更大的统一内存容量。

## 7. 面向 LLM 性能计算器的建议

若将两种平台加入 LLM 性能计算器：

1. GB10 的内存容量填写 128 GB 时，应命名为“统一系统内存”，而不是“独立显存”。
2. GB10 可直接采用的官方带宽为 273 GB/s。
3. GB10 只有 FP4 稀疏峰值 1 PFLOP 可以标注为官方公开值。
4. 不应把 1 PFLOP FP4 自动换算为 FP8、BF16 或 FP16 算力。
5. M50 可记录 100 TFLOPS bFP16 与 160 TOPS INT8，但不可把 bFP16 直接等同于标准 BF16，除非后摩官方进一步说明二者语义完全一致。
6. M50 的容量、位宽和带宽应保留“最高/Max to”限定词。
7. 两个平台的预测结果应标明精度来源与稀疏假设，避免不同口径曲线被误认为同精度比较。

## 8. 结论

- GB10 不只支持 FP4；FP4 是其公开峰值性能指标，而不是唯一支持精度。
- GB10 在官方标称内存容量和带宽上高于 M50。
- M50 的官方典型功耗明显更低，但不能据此直接得出同负载能效倍率。
- 当前官方资料不足以给出二者同精度 AI 算力倍率。
- 若需要判断具体 LLM 的 TPS、TTFT 或每瓦性能，必须进行同模型、同精度、同软件版本的实测。

## 9. 官方来源

1. [NVIDIA DGX Spark 产品规格](https://www.nvidia.com/en-eu/products/workstations/dgx-spark/)
2. [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
3. [NVIDIA CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus)
4. [NVIDIA NIM for LLM：DGX Spark 支持矩阵](https://docs.nvidia.com/nim/large-language-models/latest/deploy-on-dgx-spark.html)
5. [NVIDIA Visual GenAI NIM 模型与精度配置](https://docs.nvidia.com/nim/visual-genai/latest/models.html)
6. [NVIDIA CUTLASS 官方文档](https://docs.nvidia.com/cutlass/latest/index.html)
7. [后摩漫界 M50 官方产品页](https://houmoai.com/60/ProductType.html)

