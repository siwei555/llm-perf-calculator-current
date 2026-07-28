import type { ModelDefinition } from "../../domain/model/types";

export const qwen3_5Models: ModelDefinition[] = [
  {
    family: "qwen3.5",
    id: "qwen3.5-35b-a3b",
    displayName: "Qwen3.5-35B-A3B",
    architectureKind: "hybrid-linear-moe",
    formulaStrategyId: "hybrid-linear-moe",
    configSource: "docs/Qwen_3.5/config/Qwen3.5-35B-A3B-config.json",
    parameterSourceUrl: "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4/blob/main/config.json",
    weightSourceUrl: "https://huggingface.co/Qwen/Qwen3.5-35B-A3B-GPTQ-Int4",
    recommendedPrecision: {
      label: "GPTQ 4-bit weights / BF16 activations",
      bytesPerWeight: 0.5,
      bytesPerActivation: 2,
      bytesPerExpert: 0.5
    },
    contextLimit: 262144,
    decoderLayers: 40,
    hiddenSize: 2048,
    attentionHeads: 16,
    kvHeads: 2,
    headDim: 256,

    // —— compressed-moe 字段（不适用） ——
    qLoraRank: 0,
    oLoraRank: 0,
    oGroups: 0,
    indexHeads: 0,
    indexHeadDim: 0,
    indexTopk: 0,
    csaCompressRate: 0,
    hcaCompressRate: 0,
    moeExperts: 256,
    activeExperts: 8,
    moeIntermediateSize: 512,
    csaLayerCount: 0,
    hcaLayerCount: 0,

    // —— 通用 ——
    slidingWindow: 0,
    slidingLayerCount: 0,

    // —— dense-decoder 专用 ——
    fullAttentionLayerCount: 10,
    slidingAttentionLayerCount: 0,
    globalHeadDim: 256,
    numGlobalKeyValueHeads: 2,
    attentionKEqV: false,
    hiddenActivation: "silu",

    // —— hybrid-linear-moe 专用 ——
    linearAttentionLayerCount: 30,
    linearNumKeyHeads: 16,
    linearKeyHeadDim: 128,
    linearNumValueHeads: 32,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,

    totalParamsB: 34.66,
    totalExpertParamsB: 32.36,
    estimatedWeightsGb: 18.86
  }
];
