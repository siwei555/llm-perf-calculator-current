# Qwen3.5-35B-A3B 架构结构图（ASC）

> 基于 [`Qwen3.5-35B-A3B-config.json`](./config/Qwen3.5-35B-A3B-config.json) 配置。
> 注意：本文结合 `config.json` 与本地 Python 环境中的 `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe` / `configuration_qwen3_5_moe` 源码核对。当前本地 `transformers` 版本为 `5.9.0`，配置文件中声明的 `transformers_version` 为 `4.57.0.dev0`。
> 架构家族：`Qwen3_5MoeForConditionalGeneration`（decoder-only，多模态视觉+文本，本文仅分析文本侧）。

---

## 1. 顶层结构（Top-level）

`Qwen3_5MoeForConditionalGeneration` 继承 `Qwen3_5MoePreTrainedModel + GenerationMixin`，由以下核心部件组成：

| 组件 | 类型 | 说明 |
|------|------|------|
| `model.language_model.embed_tokens` | `nn.Embedding` | vocab=248320 → hidden=2048 |
| `model.language_model.layers` | `nn.ModuleList[Qwen3_5MoeDecoderLayer]` | 共 **40** 层 |
| `model.language_model.norm` | `Qwen3_5MoeRMSNorm` | 最终 RMSNorm（eps=1e-6） |
| `model.language_model.rotary_emb` | `Qwen3_5MoeTextRotaryEmbedding` | M-RoPE，复用给所有层 |
| `lm_head` | `nn.Linear` | hidden=2048 → vocab=248320（`tie_word_embeddings=false`） |
| `model.visual` | Vision Transformer (27层) | 视觉编码器（本文不展开） |

文本路径为：`input_ids[B, S]` → `model.language_model.embed_tokens` → 40 个 decoder 层 → `model.language_model.norm` → `lm_head` → `logits[B, S, V]`。多模态顶层还包含 `model.visual`，视觉特征会被写入文本 embedding 序列中。

### 1.1 关键超参（取自 config）

| 字段 | 值 | 含义 |
|------|-----|------|
| `hidden_size` | 2048 | 隐层维度 D |
| `num_hidden_layers` | 40 | 解码层数 L |
| `num_attention_heads` | 16 | 全注意头数 n_h（full attention 层） |
| `num_key_value_heads` | 2 | KV 头数 n_kv（GQA，full attention 层） |
| `head_dim` | 256 | 单头维度 c（full attention 层） |
| `linear_key_head_dim` | 128 | 线性注意 key 头维度 c_k^L |
| `linear_num_key_heads` | 16 | 线性注意 key 头数 n_kh^L |
| `linear_value_head_dim` | 128 | 线性注意 value 头维度 c_v^L |
| `linear_num_value_heads` | 32 | 线性注意 value 头数 n_vh^L |
| `linear_conv_kernel_dim` | 4 | 线性注意 1D 卷积核大小 |
| `full_attention_interval` | 4 | 全注意间隔（每 4 层插入 1 层全注意） |
| `moe_intermediate_size` | 512 | 单个路由专家 FFN 中间维度 I |
| `shared_expert_intermediate_size` | 512 | 共享专家 FFN 中间维度（与路由专家相同） |
| `num_experts` | 256 | 路由专家总数 E |
| `num_experts_per_tok` | 8 | top-k 路由激活专家数 k |
| `max_position_embeddings` | 262144 | 最大上下文（256K） |
| `rope_theta` | 10,000,000 | RoPE 基频 |
| `partial_rotary_factor` | 0.25 | RoPE 部分旋转因子（rope_dim / head_dim） |
| `mrope_interleaved` | true | M-RoPE 交错式（支持 3D 位置） |
| `mrope_section` | [11, 11, 10] | M-RoPE 维度分段（共 32 维） |
| `attn_output_gate` | true | 配置字段存在；源码中 full attention 的门控由 `q_proj` 的后一半输出实现，linear attention 使用独立 `in_proj_z` + gated RMSNorm |
| `rms_norm_eps` | 1e-6 | RMSNorm epsilon |
| `hidden_act` | `silu` | 激活函数（SwiGLU） |
| `mtp_num_hidden_layers` | 1 | 配置字段存在；源码通过 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` 忽略 MTP 权重，当前推理模型不实例化 MTP 模块 |
| `mlp_only_layers` | [] | 无纯 MLP 层 |
| `mamba_ssm_dtype` | `float32` | SSM 计算精度 |
| `quantization_config.bits` | 4 | GPTQ 4-bit 量化 |
| `quantization_config.group_size` | 128 | GPTQ 分组大小 |
| `attention_bias` | false | 注意力不含偏置 |
| `attention_dropout` | 0.0 | 注意力无 dropout |

### 1.2 注意力层 schedule（`layer_types`）

40 层按 `[linear, linear, linear, full] × 10` 严格排列：

| 类型 | 含义 | 层数 | 层索引 |
|------|------|------|--------|
| `linear_attention` | Gated DeltaNet 线性注意 | **30** | 0-2, 4-6, 8-10, 12-14, 16-18, 20-22, 24-26, 28-30, 32-34, 36-38 |
| `full_attention` | 标准 GQA + causal mask | **10** | 3, 7, 11, 15, 19, 23, 27, 31, 35, 39 |

即每 4 层一个"锚点"全局注意层 (`full_attention_interval=4`)，其余层用线性注意。

### 1.3 MoE schedule（所有层统一）

全部 40 层使用相同 MoE 配置：

| 类型 | 层数 | 层索引 |
|------|------|--------|
| `moe`（标准 top-k routing） | 40 | 0 … 39 |

无 Hash-MoE 层，这与 DeepSeek V4 不同。

---

## 2. 架构总览图

### 2.1 ASCII 字符图

```
                 Qwen3_5MoeForConditionalGeneration (text path)
   ┌──────────────────────────────────────────────────────────────────────────┐
   │                                                                          │
   input_ids [B, S]                                                           │
       │                                                                      │
       ▼                                                                      │
   model.language_model.embed_tokens                                          │
   nn.Embedding(248320 → 2048)                                                │
       │                                                                      │
       ▼                                                                      │
   [B, S, D=2048]                                                             │
       │                                                                      │
       ├──────────────────────────────────────────┐                           │
       │                                          │                           │
       ▼                                          ▼                           │
   model.language_model.rotary_emb       causal_mask / linear_attn_mask       │
   cos/sin [B, S, rope_dim=64]                                               │
       │                                                                      │
       ▼                                                                      │
   ┌──────────────────────────────────────────────────────┐                   │
   │  model.language_model.layers  ×  40  (see §3)         │                   │
   │                                                      │                   │
   │  L0   linear_attn + moe  ─┐                          │                   │
   │  L1   linear_attn + moe   │                          │                   │
   │  L2   linear_attn + moe   │  3 linear                │                   │
   │  L3   full_attn    + moe  │  1 full                  │                   │
   │  L4   linear_attn + moe   │                          │                   │
   │  ...                      │  pattern repeats ×10     │                   │
   │  L39  full_attn    + moe ─┘                          │                   │
   │                                                      │                   │
   │  linear_attn: 30 layers   full_attn: 10 layers       │                   │
   └──────────────────────────┬───────────────────────────┘                   │
                              │                                               │
                              ▼                                               │
                     ┌────────────────┐                                       │
                     │   RMSNorm      │  eps=1e-6                             │
                     └───────┬────────┘                                       │
                             ▼                                                │
                     ┌────────────────┐                                       │
                     │   lm_head      │  nn.Linear(2048, 248320)              │
                     │   2048 → V     │  (tie_word_embeddings = false)        │
                     └───────┬────────┘                                       │
                             ▼                                                │
                     logits [B, S, 248320]                                    │
                                                                              │
   └──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 单个 `Qwen3_5MoeDecoderLayer` 内部

采用标准 Pre-Norm 残差结构，与 DeepSeek V4 的 mHC 多流残差 **不同**：

```
hidden_states [B, S, D=2048]         (1 stream，无 HC multi-stream)
       │
       ├──► input_layernorm (RMSNorm)
       │         │
       │         ▼
       │    self_attn (linear_attn 或 full_attn, see §4-5)
       │         │
       │         ▼
       │    token mixer 内部门控
       │         │
       │         ▼
       │    residual_add ──────────────────┐
       │                                   │
       ├──► post_attention_layernorm       │
       │         │                          │
       │         ▼                          │
       │    mlp (MoE block, see §6)        │
       │         │                          │
       │         ▼                          │
       │    residual_add ──────────────────┤
       │                                   │
       ▼                                   ▼
   hidden_states [B, S, D=2048]
```

关键点：
- 单条残差流（无 HyperConnection / Sinkhorn-Knopp）
- Pre-Norm（RMSNorm 在子层之前）
- full attention 门控不是独立 `D→D` 线性层；源码中 `q_proj` 输出 `2 × n_h × head_dim`，切分为 Q 与 gate，`sigmoid(gate)` 在 `o_proj` 之前按 4096 维 attention 输出逐元素相乘。
- linear attention 使用 `in_proj_z: D→value_dim` 作为 gated RMSNorm 的 gate 输入，不使用 full attention 的 `q_proj` gate。

---

## 4. 全注意层（`full_attention`）内部

标准 GQA（Grouped Query Attention）+ RoPE + causal mask。

### 4.1 关键公式与维度

```
hidden_states [B, S, D=2048]
       │
       ├─────────────────────────────────────┐
       │                                     │
       ▼                                     ▼
   q_proj (D → 2·n_h·c = 8192)         k_proj (D → n_kv·c = 512)
       │                                     │
       ▼                                     ▼
   split: Q [B,S,16,256] + gate [B,S,4096]  [B, n_kv=2, S, c=256]
       │                                     │
       ▼                                     │
   q_norm(head_dim)                          │
       │                                     │
       ▼                                     ▼
   apply_rotary_pos_emb                  apply_rotary_pos_emb
   (partial, first rope_dim=64 dims)     (partial, first rope_dim=64 dims)
       │                                     │
       │                                ┌────┘
       │                                │
       ▼                                ▼
   ┌──────────────────────────────────────────┐
   │  Core Attention (causal)                 │
   │                                          │
   │  QK^T: Q @ K^T / √c                     │
   │  causal mask: q_i can see k_j for j ≤ i │
   │  softmax → attn_weights                  │
   │  AV: attn_weights @ V                    │
   │                                          │
   │  GQA: n_kv=2 → repeat_kv ×8 → 16 heads  │
   │  L_kv(seq) ≈ S / 2 (avg over queries)   │
   └──────────────────┬───────────────────────┘
                      │ [B, n_h=16, S, c=256]
                      ▼
              gate: attn_output *= sigmoid(gate)
                      │ [B, S, 4096]
                      ▼
              o_proj (n_h·c → D = 4096 → 2048)
                      │
                      ▼
              attn_output [B, S, D=2048]
```

### 4.2 GQA 配置

- Q heads: 16, KV heads: 2 → GQA ratio = 8:1
- `q_proj` 参数输出维度 = 2 × 16 × 256 = 8192，其中一半是 Q，另一半是门控向量
- KV 维度 = 2 × 256 = 512
- 每个 Q head 复用自己的 KV pair：`repeat_kv(kv_states, 8)`

### 4.3 Causal Mask

标准下三角 causal mask：query `t` 只可 attend `k ≤ t`。

对于长度为 S 的序列：
- 有效 KV 对数量 = S(S+1)/2 ≈ **S²/2**
- 相比无 mask 全量 S²，**因果约减半**

这在 FLOPs 计算中体现为 `causal_factor = 2`（乘在 FLOPs 公式上，等价于×2×S×S/2 = 2·S² 的实际计算量）。

---

## 5. 线性注意层（`linear_attention`）内部

采用 `Qwen3_5MoeGatedDeltaNet` 线性注意力，以随序列长度近似线性增长的 gated delta rule scan 替代 O(S²·D) 的二次 softmax 注意力。

### 5.1 关键公式与维度

源码中的线性注意模块为 `Qwen3_5MoeGatedDeltaNet`，核心是 gated delta rule，而不是一个单独的通用 `input_proj`：

```
hidden_states [B, S, D=2048]
       │
       ▼
   key_dim   = linear_num_key_heads   × linear_key_head_dim   = 16 × 128 = 2048
   value_dim = linear_num_value_heads × linear_value_head_dim = 32 × 128 = 4096
       │
       ├──► in_proj_qkv: D → 2×key_dim + value_dim = 8192
       │       then depthwise Conv1D over Q/K/V channels (conv_dim=8192, kernel=4)
       │
       ├──► in_proj_z: D → value_dim = 4096
       │       gate input for Qwen3_5MoeRMSNormGated / FusedRMSNormGated
       │
       ├──► in_proj_b: D → linear_num_value_heads = 32
       │       beta = sigmoid(b)
       │
       └──► in_proj_a: D → linear_num_value_heads = 32
               g = -exp(A_log) * softplus(a + dt_bias)
       │
       ▼
   split conv output:
     query [B,S,16,128], key [B,S,16,128], value [B,S,32,128]
     query/key repeat_interleave 到 32 个 value heads
       │
       ▼
   chunk_gated_delta_rule / recurrent_gated_delta_rule
   recurrent_state shape: [B, 32, 128, 128]
   conv_state shape: [B, 8192, 4]
       │
       ▼
   gated RMSNorm: norm(core_attn_out, z)
       │
       ▼
       out_proj (value_dim → D = 4096 → 2048)
              │ [B, S, D]
              ▼
       attn_output [B, S, D=2048]
```

### 5.2 计算复杂度对比

| 层类型 | 核心算子 | 复杂度 | 128K 时单层核心 FLOPs |
|--------|----------|--------|------------------------|
| `full_attention` | QK^T + softmax + AV | O(S²·n_h·c) | ~140.8 T |
| `linear_attention` | Conv + gated delta rule scan | O(S·n_v_heads·key_dim_head·value_dim_head) | ~0.15 T (<1%) |

> 线性注意的核心 scan + conv 远小于 full attention 的 S² 核心。该层主要 FLOPs 来自输入/输出投影和 MoE。

### 5.3 Conv + Gated Delta Rule 细节

- **Conv1D (kernel=4)**：对 Q/K/V 拼接后的 `conv_dim=8192` 做 depthwise causal conv，FLOPs ≈ `2 · 4 · S · 8192 = 0.066M·S`
- **gated delta rule scan**：cache 中 recurrent state 为 `[B, 32, 128, 128]`，粗略按 `2 · S · 32 · 128 · 128 = 1.05M·S` 估算
- 两者合计 FLOPs ≈ `1.11M·S`，128K 时约 0.15T，**远小于 full attention 核心的 8192·S²**

---

## 6. MoE Block 内部

全部 40 层共享相同 MoE 结构：256 路由专家 + 1 共享专家 + SwiGLU，top-k=8。

### 6.1 结构

```
hidden_states [B, S, D=2048]
       │
       ▼
   Router (Gate)
   weight: 256 × 2048
   logits = x @ W_route^T
   probs = softmax(logits)
   indices, weights = topk(probs, k=8)
       │
       ├──────────────────────────────┐
       ▼                              ▼
   Routed Experts (×256)          Shared Expert (×1)
   per expert:                    gate_proj: 2048 → 512
     gate_proj: 2048 → 512        up_proj:   2048 → 512
     up_proj:   2048 → 512        down_proj: 512 → 2048
     down_proj: 512 → 2048        SiLU(gate) ⊙ up
     SiLU(gate) ⊙ up
   │ per token: only k=8 active  │ always active
   │ weights = topk_probs
       │                              │
       ▼                              ▼
   routed_out × weights          sigmoid(shared_expert_gate(x)) × shared_out
       │                              │
       └──────────┬───────────────────┘
                  ▼
            combined = routed + shared
                  │
                  ▼
            mlp_output [B, S, D=2048]
```

### 6.2 Router 配置

- 打分函数：softmax + topk（标准 sparse MoE routing）
- k=8 激活专家（占总专家数 256 的 3.125%）
- 路由权重会除以 top-k 权重和做归一化：`router_top_value /= router_top_value.sum(...)`

### 6.3 激活函数

SwiGLU（SiLU gating）：
```
gate = SiLU(x @ W_gate)
up   = x @ W_up
h    = gate ⊙ up
out  = h @ W_down
```

---

## 7. KV Cache 设计

### 7.1 全注意层 KV Cache（10 层）

标准 KV cache：每层存储 K、V 各 `[B, n_kv, S_ctx, c]`。

```
M_full_kv_cache(per_layer, S_ctx)
  = B × 2(K+V) × n_kv × S_ctx × c × bytes_per_elem
  = B × 2 × 2 × S_ctx × 256 × 2   (bf16)
  = B × S_ctx × 2048  Bytes
```

S_ctx=128K 时：`B × 131072 × 2048 = B × 268.4 MB`

全模型 10 层：`B × 2.68 GB`（B=1 时）

### 7.2 线性注意层状态（30 层）

线性注意层不维护随上下文长度增长的 KV cache，而是在 `DynamicCache` 的 linear-attention layer 中维护 `conv_states` 与 `recurrent_states`：

```
M_linear_state(per_layer)
  = conv_state + recurrent_state
  = [B, conv_dim=8192, kernel=4] + [B, n_v_heads=32, key_head_dim=128, value_head_dim=128]
  ≈ 8192 × 4 × 2B + 32 × 128 × 128 × 4B
  ≈ 2.16 MB  (per layer, per batch; recurrent state 按 fp32 估算)
```

30 层合计：`B × 30 × 2.16 = B × 64.8 MB`。如果具体 kernel 以 bf16 保存 recurrent state，则约减半；但配置中 `mamba_ssm_dtype=float32`，这里采用 fp32 粗估。

### 7.3 持久 cache 总览

| 层类型 | 层数 | Cache 模式 | 单层 128K 占用 (B=1) |
|--------|------|-----------|----------------------|
| `full_attention` | 10 | KV cache (K+V) | 268.4 MB |
| `linear_attention` | 30 | Conv state + recurrent state | 2.16 MB |
| **合计** | 40 | — | **≈ 2.75 GB** |

---

## 8. RoPE 与位置编码

### 8.1 M-RoPE (Multi-Resolution RoPE)

```
rope_parameters:
  rope_type: "default"
  rope_theta: 10000000
  mrope_interleaved: true
  mrope_section: [11, 11, 10]     → 共 32 dim
  partial_rotary_factor: 0.25     → rope_dim = 64 (256 × 0.25)
```

- **M-RoPE**：`mrope_section=[11,11,10]` 表示频率半维共 32 维；源码会将 3D position ids 的 T/H/W 频率交错写回文本频率布局
- **partial RoPE**：`rope_dim = head_dim × partial_rotary_factor = 64`；源码对 Q/K 的前 64 维施加 RoPE，剩余 192 维直通
- **interleaved**：配置字段存在；本地源码通过 `apply_interleaved_mrope` 将 H/W 频率按 offset=1/2、步长 3 写入 T 频率张量

### 8.2 全注意 vs 线性注意

- **全注意层**：Q、K 均施加 M-RoPE（partial）
- **线性注意层**：`Qwen3_5MoeGatedDeltaNet` 不使用 RoPE，位置信息主要通过因果卷积和递推状态体现

---

## 9. 量化 / 部署相关

- `quantization_config.quant_method = "gptq"`, `bits=4`, `group_size=128`, `sym=true`
- `dynamic` 配置：`lm_head`、`model.language_model.embed_tokens`、匹配 `.*attn.*` 的模块、`shared_expert`、`mtp`、`visual` **不量化**；routed experts 是主要 GPTQ 4-bit 对象
- `mamba_ssm_dtype = "float32"`：SSM 内部状态用 fp32 计算
- 主精度：`bfloat16`（激活/权重）
- MTP 配置字段为 `mtp_num_hidden_layers=1`；源码忽略 `^mtp.*` 权重，当前 `Qwen3_5MoeForConditionalGeneration` 不实例化 MTP 模块
- `tie_word_embeddings=false`：输入和输出嵌入独立

---

## 10. 参数量估算

### 10.1 逐模块分解

| 模块 | 参数计算 | 数量 |
|------|----------|------|
| Embedding | V × D = 248320 × 2048 | 508.5M |
| lm_head | D × V = 2048 × 248320 | 508.5M |

**Full Attention 层（×10）：**

| 组件 | 计算 | 参数量 |
|------|------|--------|
| q_proj | D × 2 × n_h × c = 2048 × 8192 | 16.78M |
| k_proj | D × n_kv × c = 2048 × 512 | 1.05M |
| v_proj | D × n_kv × c = 2048 × 512 | 1.05M |
| o_proj | n_h × c × D = 4096 × 2048 | 8.39M |
| q_norm + k_norm | 2 × head_dim | ~0.001M |
| **单层注意力合计** | | **27.27M** |
| **10 层注意力合计** | | **272.7M** |

**Linear Attention 层（×30）：**

| 组件 | 计算 | 参数量 |
|------|------|--------|
| in_proj_qkv | D × (2×key_dim + value_dim) = 2048 × 8192 | 16.78M |
| in_proj_z | D × value_dim = 2048 × 4096 | 8.39M |
| in_proj_b + in_proj_a | 2 × D × n_v_heads = 2 × 2048 × 32 | 0.13M |
| out_proj | value_dim × D = 4096 × 2048 | 8.39M |
| conv weight | kernel × conv_dim = 4 × 8192 | 0.033M |
| A_log + dt_bias + gated norm | 32 + 32 + 128 | ~0.0002M |
| **单层合计** | | **33.72M** |
| **30 层合计** | | **1,011.5M** |

**MoE（×40 层，每层相同）：**

| 组件 | 计算 | 参数量 |
|------|------|--------|
| Router | D × E = 2048 × 256 | 0.52M |
| Routed Experts (×256) | 256 × (gate: D×I + up: D×I + down: I×D) | 256 × 3.15M = 805.3M |
| Shared Expert (×1) | gate: D×I + up: D×I + down: I×D | 3.15M |
| Shared expert gate | D × 1 = 2048 | 0.002M |
| **单层 MoE 合计** | | **808.98M** |
| **40 层 MoE 合计** | | **32,359.1M ≈ 32.36B** |

### 10.2 汇总

| 类别 | 参数量 | 占比 |
|------|--------|------|
| Embedding + lm_head | 1,017.1M | 3.0% |
| Full attention (×10) | 272.7M | 0.8% |
| Linear attention (×30) | 1,011.5M | 2.9% |
| MoE (×40) | 32,359.1M | 94.0% |
| RMSNorm / other | ~0.2M | <0.1% |
| **文本侧总计** | **≈ 34.66B ≈ 35B** | 100% |

**Naming check：35B total ✓**

### 10.3 每 Token 激活参数（A3B）

| 类别 | 计算 |
|------|------|
| 共享参数 | Embedding + lm_head + 所有 attention + shared experts + shared gates + norms ≈ 2.45B |
| 路由专家参数（per token） | 40 layers × 8 experts × 3.15M = 1.01B |
| **总激活参数** | **≈ 3.46B**（A3B 为产品命名口径，按源码逐模块粗加约高于 3B） |

---

## 11. 关键超参速查表

| 维度 | 值 |
|------|-----|
| 架构类型 | `Qwen3_5MoeForConditionalGeneration`（decoder-only，hybrid） |
| 层数 L | 40 |
| 隐藏维度 D | 2048 |
| 全注意头 n_h / KV 头 n_kv / 头维 c | 16 / 2 / 256（GQA 8:1） |
| 线性注意 key 头 / key 头维 | 16 / 128 |
| 线性注意 value 头 / value 头维 | 32 / 128 |
| 全注意层数 L_full | 10 |
| 线性注意层数 L_linear | 30 |
| 全注意间隔 | 4 |
| 路由专家 / 共享专家 / top-k | 256 / 1 / 8 |
| 专家中间维度 I | 512 |
| 词表 V | 248320 |
| 最大上下文 | 262,144（256K） |
| 激活函数 | SiLU（SwiGLU） |
| 主精度 | bfloat16 |
| 权重量化 | GPTQ 4-bit, group_size=128 |
| SSM 精度 | float32 |
| MTP | 配置字段为 1；推理类不实例化，忽略 `^mtp.*` 权重 |
| 文本侧总参数 / 激活参数粗估 | 34.66B / ~3.46B |

---

## 12. Prefill 阶段算力估算

> 取 `S = 128 × 1024 = 131072`（128K）作为示例。
> FLOPs 计数约定：1 multiply-add = 2 FLOPs；norm / RoPE / embedding / router topk 等小项忽略。
> 变量名与 config 字段对应。

### 12.1 全注意层 FLOPs 模板（10 层）

| 块 | 公式 | 变量代入 | 128K 数值 |
|----|------|----------|-----------|
| Q+gate 投影 | `2·S·D·(2·n_h·c)` | 2S·2048·8192 | **33.55M·S** |
| K 投影 | `2·S·D·n_kv·c` | 2S·2048·512 | **2.10M·S** |
| V 投影 | `2·S·D·n_kv·c` | 2S·2048·512 | **2.10M·S** |
| 输出投影 | `2·S·n_h·c·D` | 2S·4096·2048 | **16.78M·S** |
| 门控逐元素乘 | `S·n_h·c` | S·4096 | 相比 GEMM 可忽略 |

**核心注意力（因果）**：

因 causal mask，有效 QK 对 ≈ S²/2：
```
QK^T: 2 · (S²/2) · n_h · c = S² · n_h · c = 4096 · S²
AV:   2 · (S²/2) · n_h · c = S² · n_h · c = 4096 · S²
FLOPs_core = 2 · S² · n_h · c = 8192 · S²
```

> **对比**：非因果全量核心注意 FLOPs = `4·S²·n_h·c = 16384·S²`，因果约减半。
> **公式中的 causal_factor = 2** 即体现此减半效果。

**MoE（每层相同）**：
```
FLOPs_MoE = 6 · S · D · I · (k + 1)
          = 6S · 2048 · 512 · 9
          = 56.62M · S
```

**全注意层单层汇总**（S=128K=131072）：

| 块 | 128K FLOPs |
|----|------------|
| Q+gate 投影 | 4.40 T |
| K 投影 | 0.27 T |
| V 投影 | 0.27 T |
| 输出投影 | 2.20 T |
| 核心注意 (causal, S²/2) | **140.79 T** ← 大头！ |
| MoE | 7.42 T |
| **单层合计** | **155.31 T** |

### 12.2 线性注意层 FLOPs 模板（30 层）

| 块 | 公式 | 128K 数值 |
|----|------|-----------|
| in_proj_qkv | `2·S·D·(2·key_dim + value_dim)` | 2S·2048·8192 = **33.55M·S = 4.40 T** |
| in_proj_z | `2·S·D·value_dim` | 2S·2048·4096 = **16.78M·S = 2.20 T** |
| in_proj_a + in_proj_b | `2·S·D·2·n_v_heads` | 2S·2048·64 = **0.26M·S = 0.03 T** |
| Conv1D (kernel=4) | `2·kernel·S·conv_dim` | 8S·8192 = **0.066M·S ≈ 0.009 T** |
| gated delta rule scan | `2·S·n_v_heads·c_k^L·c_v^L` | 2S·32·128·128 = **1.05M·S = 0.14 T** |
| 输出投影 (out_proj) | `2·S·value_dim·D` | 2S·4096·2048 = **16.78M·S = 2.20 T** |
| MoE | `6·S·D·I·(k+1)` | **56.62M·S = 7.42 T** |
| **单层合计** | | **16.40 T** |

> 线性注意层的核心 scan + Conv 合计约 0.15 T/层。相比 full attention 的 S² 核心仍很小，主要算力来自投影和 MoE。

### 12.3 三层汇总

| 块 | 线性注意 (×30) | 全注意 (×10) | 合计 |
|----|---------------|-------------|------|
| 输入/QKV/门控投影 | 198.90 T | 49.40 T | 248.30 T |
| 输出投影 | 66.00 T | 22.00 T | 88.00 T |
| Conv + gated delta scan | 4.38 T | — | 4.38 T |
| 核心注意 (causal) | — | **1,407.90 T** | **1,407.90 T** |
| MoE | 222.60 T | 74.20 T | 296.80 T |
| **合计** | **491.88 T** | **1,553.10 T** | **2,045.00 T ≈ 2.05 PFLOPs** |

### 12.4 算力占比

```
                    128K Prefill 算力构成 (~2.05 PFLOPs)
  ┌────────────────────────────────────────────────────────────┐
  │ Full 10层 core_attn       ████████████████████████████████  68.8% │
  │ Linear 30层 moe           █████████                      10.9% │
  │ Linear 30层 input proj    █████████                       9.7% │
  │ Full 10层 moe             ███                             3.6% │
  │ Linear 30层 out_proj      ███                             3.2% │
  │ Full 10层 Q/K/V/gate proj ██                              2.4% │
  │ 其它 (conv/scan/full out_proj)                            1.4% │
  └────────────────────────────────────────────────────────────┘
```

> **核心发现**：
> 1. **全注意核心（10 层 × causal S²-attn）占 68.8%**（≈1408 TFLOPs），是绝对瓶颈
> 2. MoE 合计（40 层 × routed+shared）占 14.5%
> 3. 线性注意层（30 层）的 scan+conv 核心约 0.15 T/层，远小于投影和 MoE

### 12.5 不同输入长度对比

| 输入长度 S | Full core (10层) | Full total (10层) | Linear total (30层) | **Total Prefill** |
|-----------|------------------|-------------------|--------------------|--------------------|
| 4K (4096) | 10 × 0.137T = 1.37T | 10 × 0.59T = 5.93T | 30 × 0.51T = 15.37T | **21.30 T** |
| 32K (32768) | 10 × 8.80T = 87.96T | 10 × 12.44T = 124.38T | 30 × 4.10T = 122.99T | **247.37 T** |
| 128K (131072) | 10 × 140.74T = 1407.37T | 10 × 155.31T = 1553.06T | 30 × 16.40T = 491.95T | **2,045.01 T ≈ 2.05 P** |

> **注意**：随 S 增长，全注意核心 O(S²) 项迅速膨胀：
> - 4K 时 core attn 占比 = 1.37/21.30 = 6.4%
> - 128K 时 core attn 占比 = 1407.37/2045.01 = 68.8%
> - 若上下文到 256K（max_position_embeddings），core attn 将增至 ~5.63 PFLOPs（仅 10 层），总 prefill 粗估约 6.90 PFLOPs

---

## 13. Decode 阶段内存需求公式（128K 场景）

> 本节只讨论 decode 阶段显存。算力估算移到 §14，避免把 FLOPs 与 memory budget 混在同一节。
>
> 口径：`S_ctx = 128 × 1024 = 131072`，`B = 1`，KV cache 元素按 bf16（`e = 2B`）估算；线性注意 recurrent state 按 fp32 估算。

### 13.1 Weight 常驻显存

文本侧总参数量 `34.66B` 来自 §10 的逐模块累加。对 `Qwen3.5-35B-A3B-GPTQ-Int4`，推理框架加载后的权重常驻规模实测约 **23 GB**，因此权重显存以 load 后观测值作为主口径。

| 组件 | 参数量 | 显存口径 | 理论显存 |
|------|--------|----------|----------|
| Embedding | 0.509B | bf16 | 1.02 GB |
| lm_head | 0.509B | bf16 | 1.02 GB |
| Norms / small params | ~0.000B | bf16 | ~0.00 GB |
| Full attention (10 层) | 0.273B | GPTQ 4-bit | 0.14 GB |
| Linear attention (30 层) | 1.012B | GPTQ 4-bit | 0.51 GB |
| Routed experts (40 层 × 256 experts) | 32.212B | GPTQ 4-bit | 16.11 GB |
| Shared experts + router + shared gate | 0.147B | GPTQ 4-bit | 0.07 GB |
| **理论小计** | **34.66B** | — | **18.86 GB** |
| GPTQ 元数据 / packing / runtime 开销 | — | 实测校准 | **~4.14 GB** |
| **Weight 常驻显存** | — | load 后观测 | **~23 GB** |

```text
M_weight_resident
  ≈ N_bf16_non_gptq × 2B + N_gptq_linear × 0.5B + M_gptq_meta/runtime
  ≈ 1.02B × 2 + 33.64B × 0.5 + 4.14 GB
  ≈ 23 GB
```

说明：
- `N_bf16_non_gptq` 主要包含 `embed_tokens`、`lm_head`、norm 等，约 `1.02B`。
- `N_gptq_linear` 主要包含 attention、linear attention、routed experts、shared expert、router 等 Linear / 参数矩阵，约 `33.64B`。
- `quantization_config.dynamic` 的模块规则需要结合具体 GPTQ loader 解释；本文以推理框架实际 load 后约 23 GB 的观测值校准权重显存。

### 13.2 Cache / State 常驻显存

Decode 持久状态由两类 layer 组成：

- `full_attention` 层维护随上下文长度增长的 KV cache。
- `linear_attention` 层不维护长 KV cache，只维护固定大小的 `conv_state` 与 `recurrent_state`。

| Layer 类型 | 层数 | 常驻状态 | 单层公式 | 单层 128K 占用 (B=1) | 合计 |
|------------|------|----------|----------|----------------------|------|
| Full attention | 10 | KV cache | `B × 2(K+V) × n_kv × S_ctx × c × e` | 268.4 MB | **2.68 GB** |
| Linear attention | 30 | Conv state | `B × conv_dim × kernel × 2B` | 0.066 MB | 0.002 GB |
| Linear attention | 30 | Recurrent state | `B × n_v_heads × c_k^L × c_v^L × 4B` | 2.10 MB | 0.063 GB |
| **合计** | 40 | — | — | — | **≈ 2.75 GB** |

展开：

```text
M_full_kv(per_layer)
  = 1 × 2 × 2 × 131072 × 256 × 2
  = 268.4 MB

M_full_kv(total)
  = 10 × 268.4 MB
  = 2.68 GB

M_linear_state(per_layer)
  = 8192 × 4 × 2 + 32 × 128 × 128 × 4
  = 2.16 MB

M_linear_state(total)
  = 30 × 2.16 MB
  = 0.065 GB
```

### 13.3 单步 Decode 临时工作集

单步 decode 的临时显存主要来自 full attention kernel 对 KV 的读取/展开方式。若实现中 materialize `repeat_kv` 后的 K/V，则单个 full attention layer 的临时工作集可按下式估算：

```text
M_full_attn_tmp(per_layer)
  ≈ B × 2(K+V) × n_h × (S_ctx + 1) × c × e
  = 1 × 2 × 16 × 131073 × 256 × 2
  = 2.15 GB
```

这不是 10 层常驻显存。decoder 按层执行时，通常只形成当前 layer 的临时峰值；高效 kernel 也可能避免完整 materialize `repeat_kv`，因此实际峰值应以具体推理框架为准。

其他单步临时项较小：

| 临时项 | 规模 | 说明 |
|--------|------|------|
| Full attention `attn_weights` | `B × n_h × 1 × S_ctx`，fp32 约 8.4 MB | 远小于 repeated K/V |
| Linear attention step | MB 级以内 | 使用固定大小 recurrent/conv state |
| 当前 hidden / logits | MB 级以内 | 单 token decode 下较小 |

### 13.4 Decode 总显存预算

| 项目 | 类型 | 128K 数值 (B=1) | 是否常驻 |
|------|------|------------------|----------|
| Weight | GPTQ-Int4 load 后权重 | **~23.00 GB** | 是 |
| Full attention KV cache | 10 层 full KV | **2.68 GB** | 是 |
| Linear attention state | 30 层 conv/recurrent state | **0.065 GB** | 是 |
| **常驻小计** | Weight + cache/state | **~25.75 GB** | 是 |
| Full attention 单层临时工作集 | repeated K/V 上界估算 | **~2.15 GB** | 否 |
| **保守峰值估算** | 常驻小计 + 单层临时上界 | **~27.90 GB** | 峰值 |

结论：128K、B=1 的 decode 内存预算中，常驻显存主要由 **Weight (~23 GB)** 和 **Full KV cache (~2.68 GB)** 构成；linear attention state 只有约 **0.065 GB**。若推理 kernel materialize full attention 的 repeated K/V，单层临时峰值约再增加 **2.15 GB**。

## 14. Decode 阶段算力估算（单 token）

本节估算单 token decode FLOPs，数据来源仍按 weight 与 cache/state 区分。

**全注意层**（单 token）：
- Q/gate/K/V 投影：~54.53M
- Core attention：`2 × 1 × S_ctx × n_h × c × 2 = 2 × 131072 × 16 × 256 × 2` ≈ 2.15 G
  - QK^T: `2 × 1 × S_ctx × n_h × c = 2 × 131072 × 16 × 256 ≈ 1.07 G`
  - AV:    `2 × 1 × S_ctx × n_h × c ≈ 1.07 G`
- 输出投影：~16.78M

**线性注意层**（单 token）：
- in_proj_qkv / z / a / b + out_proj：~67.37M
- 线性注意核心（Conv + recurrent gated delta step）：约 1.1M
  - Conv update：`2 × kernel × conv_dim = 2 × 4 × 8192 ≈ 0.066M`
  - Recurrent gated delta step：`2 × n_v_heads × c_k^L × c_v^L = 2 × 32 × 128 × 128 ≈ 1.05M`

**MoE**（每层，单 token）：
- `6 × D × I × (k+1) = 6 × 2048 × 512 × 9 ≈ 56.62 M`

**Decode FLOPs 汇总（单 token）**：

| 块 | 数据来源 | 全注意 (×10) | 线性注意 (×30) | 合计 |
|----|----------|-------------|---------------|------|
| 投影 | Weight | 0.71 G | 2.02 G | 2.73 G |
| Full core attention (QK/AV over KV cache) | Cache | 21.47 G | — | 21.47 G |
| Linear core (Conv + recurrent gated delta step) | Cache / state | — | ~0.03 G | ~0.03 G |
| MoE | Weight | 0.57 G | 1.70 G | 2.26 G |
| **合计** | — | **22.75 G** | **3.75 G** | **26.50 GFLOPs/token** |

按数据来源聚合：

| 数据来源 | FLOPs/token | 说明 |
|----------|-------------|------|
| Weight | **4.99 G** | 投影 + MoE，主要读取 GPTQ/bf16 权重 |
| Cache / state | **21.50 G** | full attention 读取 KV cache；linear attention 更新 conv/recurrent state |
| **合计** | **26.50 G** | 单 token decode 粗估 |

> Decode 算力大头是全注意层读取 KV cache 做 QK^T 和 AV（~21.5 G，占 81%）。线性注意没有对历史 KV 做 `S_ctx` 长度的 QK/AV，而是通过 recurrent state 做 O(1) 的 gated delta step，因此核心计算约 `0.03 G`（30 层合计），远小于 full attention core。

---

## 15. 与 DeepSeek V4 架构差异对比

| 维度 | DeepSeek V4 Flash | Qwen3.5-35B-A3B |
|------|-------------------|-----------------|
| 残差流 | 4 条 mHC 流 + Sinkhorn-Knopp | 单条 Pre-Norm |
| 注意力类型 | 全注意（MLA + compressed） | 混合（full GQA + Gated DeltaNet） |
| 全注意 KV | Shared-KV MQA (n_kv=1) | GQA (n_kv=2) |
| 线性/压缩注意 | CSA (×21) + HCA (×20) | Gated DeltaNet (×30) |
| Q 投影 | LoRA-Q (q_a→r_q→q_b) | `q_proj` 同时产生 Q 与 gate |
| 输出投影 | GroupedLinear (o_a→o_b) | 标准线性 `o_proj` |
| MoE 专家/激活 | 256 / 6 | 256 / 8 |
| MoE 中间维度 | 2048 | 512 |
| 共享专家中间维度 | (隐含在 routed 中) | 512（与 routed 相同） |
| Prefill 瓶颈 | Indexer O(S²) CSA 核心 | Full-attn O(S²) 核心 |
| Decode Cache | Compressed + KV (complex) | 10 层 full KV + 30 层 conv/recurrent state |
| 位置编码 | YaRN (compress) + RoPE (main) | M-RoPE (partial, interleaved) |

---

## 16. 引用源

- 配置文件：`docs/Qwen_3.5/config/Qwen3.5-35B-A3B-config.json`
- 本地源码：`transformers.models.qwen3_5_moe.configuration_qwen3_5_moe`、`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe`、`transformers.cache_utils`
- 参考架构：Qwen3.5-MoE 公开技术报告 + gated delta rule / Mamba 系列 SSM 原理
- 对比参考：`docs/deepseek_v4/DeepSeek-V4-Flash.md`
