# Model Structure Page Spec

## 1. 页面目的

`Model Structure` 页面用于解释当前选中模型的结构组成，让用户能从性能结果跳回模型来源，理解 `prefill`、`decode` 和内存估算所依赖的结构参数。

页面面向内部研发和性能分析人员，重点不是渲染完整源码图，而是把影响性能计算的结构项组织成可扫描、可对比、可追溯的视图。

关联效果图：

- `docs/app/design/images/model-structure-v1.png`

## 2. 页面范围

本页负责：

- 当前模型选择
- 顶层结构总览
- attention / MoE / cache schedule 展示
- 关键配置参数表
- 与性能计算公式相关的结构字段标注
- 跳转回性能计算页

本页不负责：

- 完整源码浏览
- checkpoint 权重浏览
- 动态运行 trace
- 多模型并排 diff

## 3. 页面布局

页面复用应用左侧导航和顶部应用栏，主区域采用三段布局：

1. `模型选择与结构摘要`
2. `结构图与模块拆解`
3. `配置参数与性能关联`

桌面布局：

- 顶部为模型选择条和摘要指标
- 中部为 60 / 40 双列
- 底部为全宽参数表

移动布局：

- 所有区域纵向堆叠
- 结构图改为纵向模块卡片

## 4. 顶部模型摘要区

组件：

- `ModelSelector`
- `ArchitectureMetricStrip`
- `SourceBadgeGroup`

字段：

- `displayName`
- `decoderLayers`
- `hiddenSize`
- `attentionHeads`
- `kvHeads`
- `headDim`
- `moeExperts`
- `activeExperts`
- `contextLimit`
- `estimatedWeightsGb`

交互：

- 在模型结构页手动切换模型后，页面所有结构区域同步切换，但不反向修改性能计算页的模型选择
- 每次性能计算成功后，模型结构页同步到本次计算使用的模型
- 当前模型状态应与性能计算页共享或保持同一默认值

## 5. 中部结构区

### 5.1 左列：结构流图

用工程图方式展示：

```text
input_ids
  -> embed_tokens
  -> hidden stream expand
  -> decoder layers
       -> attention path
       -> hyper connection
       -> MoE path
  -> hc_head
  -> norm
  -> lm_head
  -> logits
```

要求：

- 每个模块显示输入输出维度
- `decoder layers` 必须显示层数和 attention schedule
- 当前影响性能估算的字段需用 `Used by calculator` 标识
- 维度、`Used by calculator` 标识与右侧 calculator path 必须使用独立网格列，禁止相互覆盖
- calculator path 在桌面端保留独立宽度；窄屏时隐藏路径括号，并把维度和标识堆叠到模块卡片下方
- calculator path 的竖排文字靠近右侧虚线放置，不得占用或覆盖左侧 `Used by calculator` 标识

### 5.2 右列：模块详情栈

卡片：

- `Attention`
  - `num_attention_heads`
  - `num_key_value_heads`
  - `head_dim`
  - `q_lora_rank`
  - `o_lora_rank`
  - `o_groups`

- `Compressed Cache`
  - `sliding_window`
  - `index_topk`
  - `compress_rate_csa`
  - `compress_rate_hca`

- `MoE`
  - `n_routed_experts`
  - `num_experts_per_tok`
  - `moe_intermediate_size`
  - `shared expert = 1`

## 6. Attention / MLP Schedule

展示为横向层分布条：

- `sliding_attention`
- `compressed_sparse_attention`
- `heavily_compressed_attention`
- `hash_moe`
- `moe`

显示规则：

- 每类显示层数、占比和层索引摘要
- Flash 显示 `2 sliding / 21 CSA / 20 HCA`
- Pro 显示 `0 sliding / 30 CSA / 31 HCA`

## 7. 参数表

参数表列：

- `Field`
- `Value`
- `Meaning`
- `Used In`

`Used In` 可取：

- `Prefill FLOPs`
- `Decode Cache`
- `Decode Temp`
- `Weight Memory`
- `Display Only`

默认展示核心字段，后续可增加搜索和分组折叠。

## 8. 空状态和错误状态

- 若模型定义缺失，显示 `Model definition unavailable`
- 若字段缺失，参数表中显示 `-`，并在 `Used In` 标记为 `Config gap`

## 9. 首版实现边界

首版使用 `src/engines/model-registry/` 中的静态模型定义，不直接解析 JSON 配置文件。

后续可增加：

- 从 `docs/deepseek_v4/config/*.json` 自动生成模型定义
- 模型结构 diff
- 源码路径链接
- Mermaid / SVG 导出

## 10. Qwen3.6-35B-A3B 展示规则

选择 `Qwen3.6-35B-A3B` 时，结构页必须从模型定义展示 40 个
decoder layers、hidden size 2048、16 attention heads、2 KV heads、
256 routed experts、8 active experts/token、MoE intermediate size 512
和 262144 context limit。结构流图需同时区分 10 个 Full GQA 层和
30 个 Gated DeltaNet 层，避免把 hybrid schedule 展示成单一注意力层。

`Qwen3.6-27B`同样展示Full GQA和Gated DeltaNet两类层，
但第三张结构卡必须显示Dense SwiGLU FFN与intermediate size 17408，
不得显示MoE experts、active experts或shared expert。
