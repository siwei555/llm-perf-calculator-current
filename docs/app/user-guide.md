# LLM Perf Calculator 使用说明

本文档适用于当前版本的 LLM Perf Calculator。工具面向 LLM 部署评估、硬件选型和性能分析，通过本地解析公式估算 Prefill、Decode 和运行时内存，不调用在线推理服务。

> 重要：所有结果都是基于模型结构、平台峰值和效率假设得到的工程估算，不是硬件实测数据。正式容量规划应使用目标运行时的 benchmark 校准效率参数。

## 1. 工具可以做什么

选择模型并输入硬件与工作负载参数后，工具可以计算和展示：

- TTFT（Time To First Token）；
- Prefill TPS；
- Initial Decode TPS；
- Average Decode TPS；
- 完整输出区间的 Decode Time；
- 权重、持久 Cache、临时工作集、运行时开销和峰值内存；
- Compute ceiling、Bandwidth ceiling 和当前瓶颈；
- 不同 Token 长度下的性能和内存趋势；
- Prefill、Decode、Memory 的公式追溯与中间量；
- 按模型实际架构拆分的 HTML 理论计算报告；
- 可供程序处理的 JSON 计算快照；
- 保存在本机的计算历史。

## 2. 当前支持的模型

| 模型家族 | 模型 | 主要架构 |
| --- | --- | --- |
| DeepSeek V4 | DeepSeek-V4-Flash | Compressed MoE，包含 Sliding、CSA、HCA 与 Sparse MoE |
| DeepSeek V4 | DeepSeek-V4-Pro | Compressed MoE，包含 CSA、HCA 与 Sparse MoE |
| Gemma 4 | Gemma-4-12B-it | Dense Decoder，Sliding/Full Attention + Dense FFN |
| Gemma 4 | google/gemma-4-26B-A4B-it | Sliding/Full Attention + routed MoE |
| Qwen3.5 | Qwen3.5-35B-A3B | Full GQA + Gated DeltaNet + MoE |
| Qwen3.6 | Qwen3.6-27B | Full GQA + Gated DeltaNet + Dense SwiGLU |
| Qwen3.6 | Qwen3.6-35B-A3B | Full GQA + Gated DeltaNet + MoE |

## 3. 页面导航

左侧导航包含四个页面，每个页面下面都有可折叠的小标题。点击小标题可以直接定位到对应区域。

### 3.1 性能计算

主工作台，用于选择模型、填写参数、执行计算、查看结果和导出报告。

### 3.2 模型结构

用于查看模型结构流图、模块参数、Attention/MLP 层级排布以及结构参数对性能的影响。

性能计算页和模型结构页的模型选择彼此独立。只有在“计算性能”成功后，模型结构页和公式说明页才会同步到本次计算使用的模型。

### 3.3 公式说明

按当前公式模型展示：

- Prefill FLOPs；
- Prefill TPS；
- Decode TPS；
- Decode Memory；
- 变量含义、数据来源和本次计算的代入结果。

### 3.4 历史记录

保存每次成功计算的模型、参数快照和结果摘要。支持：

- 按模型筛选；
- 按时间正序或倒序排列；
- 展开查看输入和结果；
- 删除单条记录；
- 清空全部记录。

历史数据只保存在当前浏览器或桌面应用的本地存储中，不会上传到服务器。清除浏览器站点数据、切换浏览器或更换设备时，历史记录不会自动迁移。

## 4. 快速完成一次计算

1. 打开左侧的“性能计算”。
2. 在“模型选择”中先选模型家族，再选具体模型。
3. 填写 Prompt Token Length 和 Decode Output Tokens。
4. 根据目标硬件填写 Compute Throughput、Memory Bandwidth 和 HBM/VRAM Capacity。
5. 核对推荐精度对应的 Bytes / Weight、Bytes / Activation 和 Bytes / Expert。
6. 设置 Compute Efficiency、Bandwidth Efficiency、Batch Size 等计算假设。
7. 点击“计算性能”。
8. 查看核心结果、性能对比、Token 趋势、公式追溯和中间量。
9. 需要查看推导报告时点击“打开 HTML 报告”；需要结构化归档时点击“导出 JSON”。

输入框改变后，页面可能仍显示上一次成功计算的结果。只有再次点击“计算性能”，结果、导出快照、模型结构页和公式说明页才会更新。

## 5. 输入参数说明

### 5.1 模型选择

模型采用“家族 → 具体模型”两段式选择。选择具体模型后，工具会自动刷新：

- 推荐的权重、激活和专家字节数；
- 默认 Compute Throughput；
- 模型最大上下文约束；
- 与模型匹配的计算公式策略。

自动填入的是推荐起点，仍可手动覆盖。切换模型会覆盖之前手动填写的精度字节数和默认算力，请在模型选择完成后再调整平台参数。

### 5.2 输入长度

### Prompt Token Length

输入 Prompt 的 token 数量。它同时用于：

- Prefill 总计算量和 TTFT；
- Decode 第一个 token 开始时的上下文长度；
- Full Attention KV Cache 和相关临时工作集估算。

### Decode Output Tokens

计划生成的 token 数量。留空时默认等于 Prompt Token Length。

工具会沿生成区间累加每一步 Decode 延迟，因此 Average Decode TPS 和 Decode Time 不是用单一上下文点直接代替整段输出。

必须满足：

```text
Prompt Token Length + Decode Output Tokens <= 当前模型 Context Limit
```

### Token Sweep Start / End / Step

用于生成 Token 趋势图。每个采样点都会重新执行公式计算，不使用前端插值。

- Start 不能大于 End；
- Step 必须大于 0；
- 当 End 大于 Start 时，Step 不能大于 `End - Start`；
- Start 和 End 不能超过模型最大上下文；
- 趋势点不能超过 500 个，超过时需要增大 Step。

### 快捷输入

先点击一个长度输入框，再点击 4K、8K、32K、128K 或 1M，即可把数值填入当前输入目标。超过当前模型 Context Limit 的快捷值会被禁用；Token Sweep Step 不受 Context Limit 限制。

### 5.3 平台参数

| 参数 | 含义 | 使用建议 |
| --- | --- | --- |
| Compute Throughput (TFLOPS) | 目标精度下的平台峰值算力 | 使用与实际计算内核精度相符的值，不要直接套用其他精度峰值 |
| Memory Bandwidth (GB/s) | 峰值显存/内存带宽 | 填写设备公开规格或实测带宽上限 |
| HBM / VRAM Capacity (GB) | 可供模型使用的内存容量 | 多卡场景应按实际并行和权重切分方式理解，不能简单假设所有容量都可共享 |
| Bytes / Weight | 普通权重每参数字节数 | FP8 通常为 1，BF16 通常为 2 |
| Bytes / Activation | 激活与 Cache 元素字节数 | BF16 通常为 2；修改它会影响 Cache 和临时工作集 |
| Bytes / Expert | 专家权重每参数字节数 | FP4 通常为 0.5，FP8 通常为 1，BF16 通常为 2 |

界面在 Compute Throughput 输入框下方给出 `FP8 建议：248 TFLOPS`。精度字段下方保留通用换算提示，例如 `1=FP8, 2=BF16` 和 `0.5=FP4, 1=FP8`。选择 Qwen3.6-35B-A3B 后如果需要评估 FP8 量化，应根据目标运行时手动调整精度字节数；FP8 不再作为一个独立模型出现在模型列表中。

当前默认 Compute Throughput 规则为：推荐精度标签包含 FP8 时使用 248 TFLOPS，其他模型使用 124 TFLOPS。该规则是方便使用的默认口径，混合精度模型仍应按实际执行 kernel 和硬件规格手动校准。

### 5.4 计算假设

### Batch Size

并行处理的序列数量。Batch Size 会影响 Cache、临时工作集和部分内存流量估算。

### Compute Efficiency

峰值算力的有效利用率：

```text
Effective Compute = Compute Throughput × Compute Efficiency
```

默认值为 0.4。建议使用 0～1 的比例，并通过目标平台实测反推校准。

### Bandwidth Efficiency

峰值带宽的有效利用率：

```text
Effective Bandwidth = Memory Bandwidth × Bandwidth Efficiency
```

默认值为 0.6。结果为 compute-bound 时，调整它可能暂时不改变最终 TPS；这不代表带宽没有参与计算。

### Prefill Cache Traffic Factor

表示 Prefill 阶段估算为实际流量的持久 Cache 容量比例：

```text
B_prefill = B_weights + M_cache × Prefill Cache Traffic Factor
```

默认值为 0.10，允许范围为 0～1。这是可校准的工程假设，不是模型固有参数。

### Runtime Overhead

位于“内存拆解”卡片中，可直接编辑。用于表示框架、CUDA context、allocator 和 kernel workspace 等额外内存，默认值为 4 GB，必须大于或等于 0。

### 显示开关

- Show Intermediate Metrics：显示 FLOPs、ceiling、cache 和权重内存等中间量；
- Show Formula Trace：显示 Prefill、Decode 和 Memory 的逐项公式及代入结果。

## 6. 结果解读

### 6.1 核心结果

| 指标 | 含义 |
| --- | --- |
| TTFT | 完成 Prompt Prefill 并准备生成首 token 的估算时间 |
| Prefill TPS | Prompt 阶段平均处理 token 的速度 |
| Peak Runtime Memory | Decode 输出区间末端的峰值运行内存 |
| Initial Decode TPS | Prompt 完成后生成第一个 token 时的瞬时吞吐 |
| Average Decode TPS | 完整 Decode Output Tokens 区间按逐步延迟累计得到的平均吞吐 |
| Decode Time | 生成全部 Decode Output Tokens 的估算总时间 |

### Compute-bound 与 Bandwidth-bound

工具采用 Roofline 形式：

```text
TPS = min(Compute ceiling, Bandwidth ceiling)
```

- compute-bound：有效算力上限较低，算力是当前主要约束；
- bandwidth-bound：有效带宽上限较低，权重或 Cache 流量是当前主要约束。

改变非主导因素时，最终 TPS 可能不发生变化。例如 compute-bound 状态下提高带宽效率，只有当带宽 ceiling 降到或升到足以改变最小值时才会影响 TPS。

### 6.2 内存拆解

```text
Peak Runtime Memory
  = Weights
  + Persistent Decode Cache / State
  + Peak Temporary Working Set
  + Runtime Overhead
```

- Weights：普通权重和专家权重的常驻内存；
- Persistent Decode Cache：KV Cache 或线性/压缩注意力的 recurrent state；
- Peak Temporary Working Set：单步 Decode 的临时工作集峰值；
- Runtime Overhead：可编辑的运行时额外开销；
- Estimated Total：上述项目之和。

如果估算值超过 HBM/VRAM Capacity，页面会显示内存不足提示，但不会人为修改 TPS 结果。此时应重新评估并行切分、量化、上下文长度或目标硬件。

### 6.3 性能对比和趋势图

性能对比表用于并列查看 Prefill 与 Decode 的计算需求、内存流量和 ceiling。

Token 趋势图支持：

- Prefill & Decode TPS；
- TTFT；
- Total Runtime Memory；
- Bottleneck 背景；
- 数据点显示。

鼠标悬停到数据点可以查看该 Token Length 下的完整指标和 Prefill/Decode 瓶颈。

### 6.4 公式追溯和中间量

公式追溯按 Prefill、Decode、Memory 分类，显示：

- 公式名称；
- 代入表达式；
- 当前求值结果；
- 可用时提供模型参数或权重来源链接。

中间量表通过 `config`、`derived`、`formula` 标记数据来源，适合检查输入是否使用了预期口径。

## 7. HTML 报告

点击“打开 HTML 报告”会在浏览器新页面中直接呈现自包含报告，不再下载 `.html` 文件。报告基于最近一次成功计算的快照，不读取尚未重新计算的表单草稿。如果浏览器拦截弹出页面，需要允许当前站点打开新窗口后重试。

报告可以：

- 在独立页面中浏览完整计算分块；
- 通过浏览器打印为横向 PDF；
- 通过浏览器“另存为”按需保存完整网页；
- 无需安装 Excel。

报告按模型的 `formulaStrategyId` 使用不同分栏：

| 架构策略 | 报告运算分块 |
| --- | --- |
| hybrid-linear-dense | Dense FFN、Full GQA、Gated DeltaNet |
| hybrid-linear-moe | Active + Shared MoE、Full GQA、Gated DeltaNet |
| dense-decoder-transformer | Dense FFN、Sliding-window Attention、Full Attention |
| dense-decoder-moe | Active + Shared MoE、Sliding-window Attention、Full Attention |
| deepseek-v4-compressed-moe | Sparse MoE、Sliding Attention、CSA、HCA，并展开 Compressor 和 Indexer |

彩色 Type 分块表示主要运算模块；块内逐行列出投影、Attention Core、Conv1D、Delta Scan、Compressor、Indexer 等实际参与计算的算子。报告中的理论 FLOPs 不是 profiler 采集的 kernel 时间。

## 8. JSON 导出

点击“导出 JSON”会下载结构化计算快照，主要包含：

- `schemaVersion` 和导出元数据；
- 独立单位说明；
- 完整模型定义；
- Platform 和 Workload；
- Summary、Comparison 和 Memory Breakdown；
- Prefill/Decode Projection；
- Intermediate Metrics；
- Formula Trace；
- Token Trend。

数值保持 JSON number，单位不拼接进数值字符串。JSON 直接由计算快照生成，不经过 HTML 或其他导出格式反向解析。当前版本只支持导出，不支持从 JSON 恢复计算配置。

## 9. 历史记录使用

每次“计算性能”成功后会自动新增一条历史记录。仅修改输入不会新增记录。

记录中包括：

- 模型和时间；
- Prompt、Decode 和 Token Sweep；
- 算力、带宽、效率和精度；
- TTFT、Prefill TPS、Initial/Average Decode TPS；
- Decode Time、Final Decode Context 和 Peak Runtime Memory；
- Prefill/Decode 瓶颈和内存是否满足容量。

删除单条记录或清空全部记录前，工具会要求确认。删除后无法从工具内恢复。

## 10. 默认配置

执行“重置”后恢复为：

| 参数 | 默认值 |
| --- | ---: |
| 模型 | DeepSeek-V4-Flash |
| Compute Throughput | 248 TFLOPS（推荐精度标签包含 FP8） |
| Memory Bandwidth | 273 GB/s |
| HBM / VRAM Capacity | 128 GB |
| Compute Efficiency | 0.4 |
| Bandwidth Efficiency | 0.6 |
| Prefill Cache Traffic Factor | 0.1 |
| Batch Size | 1 |
| Runtime Overhead | 4 GB |
| Bytes / Weight | 1 |
| Bytes / Activation | 2 |
| Bytes / Expert | 0.5 |
| Prompt Token Length | 131,072 |
| Decode Output Tokens | 留空，计算时采用 131,072 |
| Token Sweep | 4,096 ～ 131,072，Step 4,096 |

注意：在默认 DeepSeek-V4-Flash 配置下，Prompt 131,072 加上留空后解析得到的 Decode 131,072，合计 262,144，没有超过该模型 1,048,576 的上下文上限。

## 11. 常见问题

### 修改参数后，为什么报告还是旧值？

导出和结果都使用最近一次成功计算的快照。修改参数后需要重新点击“计算性能”。

### 为什么提高带宽后 TPS 没变化？

当前结果可能是 compute-bound。工具始终同时计算 Compute ceiling 和 Bandwidth ceiling，并取较小者。

### 为什么提高算力后 Decode TPS 改善不明显？

Decode 经常受到权重读取和 Cache 流量限制。如果 Bandwidth ceiling 更低，提高算力不会成为最终 TPS 的主导因素。

### Initial Decode TPS 和 Average Decode TPS 为什么不同？

Initial Decode TPS 对应 Prompt 末端的第一个生成 token；Average Decode TPS 覆盖完整输出区间。随着上下文增长，Cache 流量和临时内存可能增加，因此平均值通常更低。

### 为什么模型切换后 Compute Throughput 会改变？

工具会根据模型推荐精度标签刷新默认算力：包含 FP8 时为 248 TFLOPS，其他为 124 TFLOPS。该值仍需按实际硬件修正。

### HTML 报告能否作为实测报告？

不能。HTML 报告记录的是理论算子拆解和当前平台假设下的估算结果。它适合设计评审和口径追溯，不替代 profiler 或端到端 benchmark。

### 多卡容量能否直接相加？

不能一概而论。是否能合并使用取决于 tensor parallel、pipeline parallel、expert parallel、KV Cache 分片和运行时实现。当前工具输入的是计算口径中的可用容量，用户需要按实际部署方式换算。

## 12. 建议使用流程

用于正式分析时，建议保留以下闭环：

1. 确认模型 config 和推荐精度；
2. 输入目标平台公开峰值；
3. 使用初始效率假设完成第一次估算；
4. 在目标运行时执行代表性 Prompt/Decode benchmark；
5. 根据实测结果校准 Compute Efficiency、Bandwidth Efficiency 和 Runtime Overhead；
6. 重新计算并导出 HTML 与 JSON；
7. 将模型版本、硬件拓扑、运行时版本和并行策略与报告一同归档。
