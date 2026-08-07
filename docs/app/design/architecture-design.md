# LLM Perf Calculator Architecture Design

## 1. 目标

本应用是一个本地运行的 LLM 性能估算工具，用于在选定模型、输入长度和平台参数后，估算：

- `TTFT`
- `Prefill TPS`
- `Decode TPS`
- `Runtime Memory`
- token 长度变化趋势
- 模型结构和公式追溯

首版支持 `DeepSeek V4` 家族，并已扩展接入 `Gemma 4`、`Qwen3.5` 和 `Qwen3.6` family。Qwen3.6 当前包含 `Qwen/Qwen3.6-35B-A3B` 与其官方 FP8 版本，两者复用 `hybrid-linear-moe` 策略，并通过模型级推荐精度区分默认权重显存口径。

## 2. 技术栈

- 前端：`React + Vite + TypeScript`
- 桌面：预留 `Tauri`
- 计算位置：浏览器本地计算
- 数据来源：首版使用静态模型定义，后续可从模型 `config.json` 生成

## 3. 目录职责

```text
src/
├── app/
│   ├── layouts/
│   ├── routes/
│   └── styles/
├── components/
├── domain/
│   ├── model/
│   ├── performance/
│   ├── platform/
│   └── workload/
├── engines/
│   ├── formula-strategies/
│   ├── model-registry/
│   └── performance-calculator/
├── features/
│   └── performance-calculator/
└── pages/
```

核心分层：

- `domain/` 定义稳定业务类型。
- `engines/model-registry/` 注册模型静态定义。
- `features/performance-calculator/services/` 承载当前性能计算公式。
- `features/performance-calculator/state/` 提供全局 calculator 状态。
- `pages/` 只做页面装配，不直接写公式逻辑。

## 4. 页面状态设计

三页共用 `CalculatorProvider`：

- `性能计算`
- `模型结构`
- `公式说明`

共享状态包括：

- 当前模型：`state.modelId`
- 平台参数：`state.platform`
- 输入长度：`state.workload`
- 展示开关：`state.view`
- 当前计算结果：`result`

这保证了：

- 性能计算、模型结构和公式说明分别维护独立的手动模型选择。
- 只有成功执行一次性能计算后，模型结构与公式说明才同步到本次计算使用的模型。
- 在性能计算页修改平台参数，公式说明页使用同一组参数。
- 模型结构页可独立浏览其他模型；每次性能计算成功后同步展示本次计算模型。

相关文件：

- `src/features/performance-calculator/state/CalculatorProvider.tsx`
- `src/features/performance-calculator/state/useCalculatorState.ts`

## 5. 模型注册设计

模型定义类型在：

- `src/domain/model/types.ts`

当前关键字段：

```ts
type ModelDefinition = {
  family: string;
  id: string;
  displayName: string;
  architectureKind:
    | "compressed-moe"
    | "dense-decoder"
    | "dense-decoder-moe"
    | "hybrid-linear-moe";
  formulaStrategyId:
    | "deepseek-v4-compressed-moe"
    | "dense-decoder-transformer"
    | "dense-decoder-moe"
    | "hybrid-linear-moe";
  configSource?: string;
  recommendedPrecision: {
    label: string;
    bytesPerWeight: number;
    bytesPerActivation: number;
    bytesPerExpert: number;
  };
  contextLimit: number;
  decoderLayers: number;
  hiddenSize: number;
  attentionHeads: number;
  kvHeads: number;
  headDim: number;
  qLoraRank: number;
  oLoraRank: number;
  oGroups: number;
  indexHeads: number;
  indexHeadDim: number;
  indexTopk: number;
  slidingWindow: number;
  csaCompressRate: number;
  hcaCompressRate: number;
  moeExperts: number;
  activeExperts: number;
  moeIntermediateSize: number;
  csaLayerCount: number;
  hcaLayerCount: number;
  slidingLayerCount: number;
  estimatedWeightsGb: number;
};
```

当前已注册模型：

- `src/engines/model-registry/deepseekV4Models.ts`
- `src/engines/model-registry/gemma4Models.ts`
- `src/engines/model-registry/qwen3_5Models.ts`
- `src/engines/model-registry/qwen3_6Models.ts`
- `src/engines/model-registry/qwen3_5Models.ts`

统一入口：

- `src/engines/model-registry/index.ts`

## 6. 公式策略设计

当前实际实现的公式策略包括：

- `deepseek-v4-compressed-moe`
- `dense-decoder-transformer`
- `dense-decoder-moe`
- `hybrid-linear-moe`
- `hybrid-linear-dense`

`Qwen3.6-35B-A3B` 使用 `hybrid-linear-moe`，不得按普通 GQA
decoder 或纯线性注意力模型计算。策略按 10 个 Full GQA 层与 30 个
Gated DeltaNet 层分别计算后汇总；MoE 项按 8 个 routed experts 加
1 个 shared expert 计算。持久缓存由 Full GQA KV cache 与线性层
conv/recurrent state 共同组成。模型事实、来源和 128K 验收基线记录在
`data/models/Qwen_3.6/Qwen3.6-35B-A3B.json`。

FP8 checkpoint 仍复用该策略，但不作为独立逻辑模型注册。量化只通过平台精度配置影响权重
显存和带宽流量：weights/expert weights 为 1 byte，activation 与
Full KV cache 为 2 bytes，Gated DeltaNet recurrent state 仍按 FP32。
其已确认事实源为
`data/models/Qwen_3.6/Qwen3.6-35B-A3B-FP8.json`。

`hybrid-linear-dense`复用Full GQA和Gated DeltaNet的attention计算，
但每层FFN使用`6·S·D·I`的dense SwiGLU公式，不使用专家、top-k或
active-expert带宽口径。当前对应逻辑模型为`Qwen3.6-27B`，
事实源为`data/models/Qwen_3.6/Qwen3.6-27B-FP8.json`。

当前计算实现位置：

- `src/features/performance-calculator/services/performanceCalculator.ts`

HTML 理论计算报告由
`src/features/performance-calculator/services/performanceHtmlReporter.ts`
直接根据 `ModelDefinition + CalculationSnapshot + PerformanceResult` 生成。
报告按 `formulaStrategyId` 穷举选择专用分块模板：hybrid-linear 模板按
Dense/MoE FFN、Full GQA、Gated DeltaNet 拆分；dense-decoder 模板按
Dense/MoE FFN、Sliding-window Attention、Full Attention 拆分；DeepSeek V4
compressed-MoE 模板按 Sparse MoE、Sliding、CSA、HCA 拆分，并在对应块中
展示 Compressor 和 Indexer。未知策略直接报错，防止套用错误算子口径。
报告为自包含 HTML，不依赖 ExcelJS。

该策略包含：

- DeepSeek V4 compressed attention schedule
- CSA / HCA / sliding attention prefill FLOPs
- MoE FLOPs
- decode weight bytes per token
- decode cache bytes per token
- persistent decode cache memory
- single-step temp peak memory
- runtime overhead

注意：未知 `formulaStrategyId` 会抛错，避免未实现策略被错误套用到其他模型上。

`dense-decoder-moe` 用于 Gemma 4 这类 sliding/full attention + routed MoE FFN 架构：

- Prefill 复用 dense decoder 的 sliding/full attention FLOPs 拆解。
- FFN FLOPs 使用 `6 * S * D * moe_intermediate_size * (activeExperts + 1)`。
- Decode 权重读取按非专家全量 + active expert fraction 估算。
- KV cache 分为 sliding window cache 和 full attention cache。

## 7. 添加新模型的推荐流程

添加一个新模型应分两步：

1. 添加模型定义。
2. 添加或复用公式策略。

不要只把模型塞进下拉框。若公式策略不匹配，性能结果会失真。

## 8. 示例：添加 `google/gemma-4-12B-it`

这里以 `google/gemma-4-12B-it` 为例说明接入流程。具体数值必须以该模型的官方 `config.json` 或本地解析结果为准，不应手填猜测值。

### 8.1 新增模型定义文件

建议新增：

```text
src/engines/model-registry/gemmaModels.ts
```

示例结构：

```ts
import type { ModelDefinition } from "../../domain/model/types";

export const gemmaModels: ModelDefinition[] = [
  {
    family: "google-gemma",
    id: "google-gemma-4-12b-it",
    displayName: "google/gemma-4-12B-it",
    architectureKind: "dense-decoder",
    formulaStrategyId: "dense-decoder-transformer",
    configSource: "data/models/google-gemma/gemma-4-12b-it-config.json",

    contextLimit: 0,
    decoderLayers: 0,
    hiddenSize: 0,
    attentionHeads: 0,
    kvHeads: 0,
    headDim: 0,

    qLoraRank: 0,
    oLoraRank: 0,
    oGroups: 1,
    indexHeads: 0,
    indexHeadDim: 0,
    indexTopk: 0,
    slidingWindow: 0,
    csaCompressRate: 0,
    hcaCompressRate: 0,

    moeExperts: 0,
    activeExperts: 0,
    moeIntermediateSize: 0,
    csaLayerCount: 0,
    hcaLayerCount: 0,
    slidingLayerCount: 0,

    estimatedWeightsGb: 0
  }
];
```

说明：

- 上面 `0` 是占位，不可作为真实结果使用。
- Dense decoder 模型不应填写 DeepSeek V4 的 CSA / HCA 语义值。
- 如果字段对 dense 模型无意义，首版可填 `0`，但更长期应把 `ModelDefinition` 拆成 discriminated union，区分 `compressed-moe` 和 `dense-decoder`。

### 8.2 注册模型

修改：

```text
src/engines/model-registry/index.ts
```

加入：

```ts
import { gemmaModels } from "./gemmaModels";

export const modelRegistry = [
  ...deepseekV4Models,
  ...gemmaModels
];
```

### 8.3 添加公式策略

`Gemma` 这类 dense decoder 模型不应使用 `deepseek-v4-compressed-moe` 公式。需要新增：

```text
src/engines/formula-strategies/denseDecoderTransformer.ts
```

建议公式口径：

```text
Prefill FLOPs
  ~= L * (
       QKV projection
     + attention QK/AV
     + output projection
     + MLP
   )

Decode bytes/token
  ~= M_weights
   + KV cache traffic per token

Decode memory
  ~= M_weights
   + KV cache
   + temp peak
   + runtime overhead
```

典型 dense decoder 字段来源：

- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim` 或 `hidden_size / num_attention_heads`
- `max_position_embeddings`
- `torch_dtype`
- quantization config

### 8.4 改造计算分发

当前 `performanceCalculator.ts` 只实现 DeepSeek V4。后续应把入口改成策略分发：

```ts
switch (model.formulaStrategyId) {
  case "deepseek-v4-compressed-moe":
    return calculateDeepseekV4(model, platform, workload);
  case "dense-decoder-transformer":
    return calculateDenseDecoder(model, platform, workload);
}
```

推荐目标结构：

```text
src/engines/formula-strategies/
├── deepseekV4CompressedMoe.ts
├── denseDecoderTransformer.ts
└── index.ts
```

### 8.5 页面适配

新增模型后需要检查：

- 性能计算页模型下拉是否显示新模型。
- 模型结构页是否能合理展示 dense 模型字段。
- 公式说明页是否显示 dense strategy 的公式 trace。
- 无意义字段不要显示为 DeepSeek V4 的 CSA / HCA 语义。

短期做法：

- 页面根据 `architectureKind` 条件展示结构卡片。

长期做法：

- 为不同架构提供结构展示 adapter：

```text
src/features/model-structure/services/
├── compressedMoeStructureAdapter.ts
└── denseDecoderStructureAdapter.ts
```

## 9. 新模型接入检查清单

- 已保存原始 `config.json` 到 `data/models/<family>/`
- 已新增模型定义文件
- 已加入 `modelRegistry`
- 已确认 `formulaStrategyId`
- 若是新架构，已新增公式策略
- 已跑 `npm run build`
- 已检查性能计算页、模型结构页、公式说明页
- 已更新 `docs/app/design/architecture-design.md`

## 10. 当前架构限制

当前代码为了快速支持 DeepSeek V4，`ModelDefinition` 仍包含一些 DeepSeek V4 特有字段，例如：

- `csaLayerCount`
- `hcaLayerCount`
- `csaCompressRate`
- `hcaCompressRate`
- `indexTopk`

这对 `Gemma` 这类 dense 模型不是理想抽象。后续接入第二个模型族时，建议把 `ModelDefinition` 拆成：

```ts
type ModelDefinition =
  | CompressedMoeModelDefinition
  | DenseDecoderModelDefinition;
```

这样可以避免 dense 模型被迫填写无意义字段。
