# Formula Notes Page Spec

> Dense Decoder MLP formulas must define `I_layer` explicitly. Base/independent-KV layers use `I_layer = I`; KV-sharing layers use `I_layer = 2I` only when the model enables double-wide MLP. The current-model formula and trace must show base/shared layer counts, actual widths, and separate FLOPs contributions rather than leaving `I_layer` undefined.

> Models with Per-Layer Embeddings must list `F_PLE`, `L`, and `P` in the formula variable table. `F_PLE` includes the global `D -> L×P` projection plus every layer's `D -> P` gate and `P -> D` projection; `P` comes from `hidden_size_per_layer_input`.

## 1. 页面目的

`Formula Notes` 页面用于解释性能计算页中的公式口径，让用户能从结果值追溯到公式、变量定义、模型字段和平台输入。

来自性能计算页的追溯链接必须携带所属公式板块和小公式目标。页面打开后自动展开所属板块，并精确滚动到对应的小公式卡片，而不是停留在板块标题处。

Gemma 4 使用的 `dense-decoder-transformer` 与 `dense-decoder-moe` 策略必须在 Decode 板块逐项展示与计算结果相同的追溯行，包括可见长度、cache/权重/总流量、每 token FLOPs、两个吞吐上限和最终 Effective Decode TPS。

页面重点是“可审计”，不是写成论文式长文。用户应能快速定位：

- `TTFT` 如何得到
- `Prefill TPS` 如何受算力和带宽限制
- `Decode TPS` 如何受 cache / weight / bandwidth 限制
- `Decode` 阶段内存为什么必须包含 weight

关联效果图：

- `docs/app/design/images/formula-notes-v1.png`

## 2. 页面范围

本页负责：

- 公式目录
- prefill 算力公式
- decode 性能公式
- decode 内存公式
- 符号表
- 与当前模型参数的代入示例

本页不负责：

- 完整模型结构图
- 公式编辑器
- 自动证明或 notebook 计算

## 3. 页面布局

页面复用应用左侧导航和顶部应用栏，主内容采用两列工作台：

- 左侧：模型选择与可展开的公式章节；公式正文和符号表收纳在对应标题下
- 右侧：当前模型代入摘要

桌面布局：

- `minmax(0, 1fr) / 340px`

窄屏布局：

- 两列改为单列
- 当前模型代入摘要下移

## 4. 章节目录

章节：

- `Prefill FLOPs`
- `Prefill TPS`
- `Decode TPS`
- `Decode Memory`
- `Symbol Table`

交互：

- 点击章节标题在标题下展开或收起对应内容
- 页面首次进入时所有章节默认收起
- 支持同时展开多个章节，便于对照公式

## 5. 公式正文

每个公式块包含：

- 标题
- 公式表达式
- 变量解释
- 当前模型代入值
- 结果单位
- 适用阶段

### 5.1 Prefill FLOPs

核心公式：

```text
F_prefill
  = N_sliding * F_sliding
  + N_csa * F_csa
  + N_hca * F_hca
```

关键子项：

```text
F_Q = 2 * S * (D * r_q + r_q * n_h * c)
F_core = 4 * S * L_kv * n_h * c
F_indexer_attn = S^2 * n_h_I * c_I / m_csa
F_moe = 6 * S * D * I * (k + 1)
```

### 5.2 Prefill TPS

```text
TPS_prefill
  = min(
      S * effective_compute / F_prefill,
      S * effective_bandwidth / B_prefill
    )
```

其中：

```text
effective_compute = platform_tflops * compute_efficiency
effective_bandwidth = memory_bandwidth * bandwidth_efficiency
```

### 5.3 Decode TPS

```text
TPS_decode
  = min(
      decode_compute_ceiling,
      decode_bandwidth_ceiling
    )
```

首版实现采用工程近似，后续需要继续细化逐算子 decode FLOPs。

### 5.4 Decode 阶段内存需求公式

必须单独成章，且必须考虑权重：

```text
M_decode_total
  ~= M_weights
   + M_decode_cache
   + M_decode_tmp_peak
   + M_runtime_overhead
```

说明：

- `M_weights` 是模型权重常驻显存，decode 阶段不能忽略
- `M_decode_cache` 是跨 token 持续存在的 cache 状态
- `M_decode_tmp_peak` 是单步 attention / matmul 的瞬时工作集峰值
- `M_runtime_overhead` 是框架、allocator、kernel workspace 的额外开销

## 6. 符号表

列：

- `Symbol`
- `Meaning`
- `Source`
- `Example`

示例：

- `S`: token length，来源 workload
- `D`: hidden size，来源 model config
- `n_h`: attention heads，来源 model config
- `c`: head dim，来源 model config
- `m_csa`: CSA compress rate，来源 model config
- `M_weights`: weight memory，来源 model definition / estimate

## 7. 当前模型代入区

右侧固定显示当前模型摘要：

- 模型名
- `F_prefill` 当前估算
- `M_weights`
- `M_decode_cache`
- `M_decode_tmp_peak`
- `M_decode_total`

首版可以使用默认模型 `DeepSeek-V4-Flash`，后续与全局模型选择状态打通。

## 8. 首版实现边界

首版采用静态公式说明 + 当前模型静态代入摘要。

后续可增加：

- 从性能计算页带入当前参数
- 点击公式变量高亮相关输入字段
- 公式块复制
- 公式版本号和来源文档链接

## 9. Qwen3.6 hybrid 公式说明

`Qwen/Qwen3.6-35B-A3B` 的说明页按三组解释：

- Prefill：Full GQA 层、Gated DeltaNet 层和 top-8 + shared MoE。
- Decode：Full KV 读取、线性 recurrent state 更新和 active expert
  weight traffic。
- Memory：BF16 model weights、Full KV cache、linear conv/recurrent
  state、临时工作集和 runtime overhead。

具体变量、推导来源及忽略项以
`docs/Qwen_3.6/Qwen3.6-35B-A3B.md` 为准。

FP8版本的公式操作数与Base相同；公式说明需要额外标注E4M3动态量化、
128×128权重块、官方非量化模块清单，以及KV cache为BF16、循环状态
为FP32。详细依据见
`docs/Qwen_3.6/Qwen3.6-35B-A3B-FP8.md`。

`Qwen3.6-27B`的Prefill追溯必须把Full GQA、Gated DeltaNet
和Dense SwiGLU分开；Decode追溯使用全部dense权重流量，不显示active
expert fraction。详细依据见`docs/Qwen_3.6/Qwen3.6-27B-FP8.md`。
