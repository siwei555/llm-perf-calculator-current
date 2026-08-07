import type { ModelDefinition } from "../../domain/model/types";

const sharedQwen36Definition = {
  family: "qwen3.6",
  architectureKind: "hybrid-linear-moe",
  formulaStrategyId: "hybrid-linear-moe",
  contextLimit: 262144,
  decoderLayers: 40,
  hiddenSize: 2048,
  attentionHeads: 16,
  kvHeads: 2,
  headDim: 256,

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

  slidingWindow: 0,
  slidingLayerCount: 0,

  fullAttentionLayerCount: 10,
  slidingAttentionLayerCount: 0,
  globalHeadDim: 256,
  numGlobalKeyValueHeads: 2,
  attentionKEqV: false,
  hiddenActivation: "silu",

  linearAttentionLayerCount: 30,
  linearNumKeyHeads: 16,
  linearKeyHeadDim: 128,
  linearNumValueHeads: 32,
  linearValueHeadDim: 128,
  linearConvKernelDim: 4,

  totalParamsB: 34.66,
  totalExpertParamsB: 32.36
} satisfies Partial<ModelDefinition>;

export const qwen3_6Models: ModelDefinition[] = [
  {
    family: "qwen3.6",
    id: "qwen3.6-27b-fp8",
    displayName: "Qwen3.6-27B",
    architectureKind: "hybrid-linear-dense",
    formulaStrategyId: "hybrid-linear-dense",
    configSource: "docs/Qwen_3.6/config/Qwen3.6-27B-FP8-config.json",
    parameterSourceUrl: "https://huggingface.co/Qwen/Qwen3.6-27B-FP8/blob/main/config.json",
    weightSourceUrl: "https://huggingface.co/Qwen/Qwen3.6-27B-FP8",
    recommendedPrecision: {
      label: "FP8 weights / BF16 activations",
      bytesPerWeight: 1,
      bytesPerActivation: 2,
      bytesPerExpert: 1
    },
    contextLimit: 262144,
    decoderLayers: 64,
    hiddenSize: 5120,
    attentionHeads: 24,
    kvHeads: 4,
    headDim: 256,

    qLoraRank: 0,
    oLoraRank: 0,
    oGroups: 0,
    indexHeads: 0,
    indexHeadDim: 0,
    indexTopk: 0,
    csaCompressRate: 0,
    hcaCompressRate: 0,
    moeExperts: 0,
    activeExperts: 0,
    moeIntermediateSize: 17408,
    intermediateSize: 17408,
    csaLayerCount: 0,
    hcaLayerCount: 0,

    slidingWindow: 0,
    slidingLayerCount: 0,

    fullAttentionLayerCount: 16,
    slidingAttentionLayerCount: 0,
    globalHeadDim: 256,
    numGlobalKeyValueHeads: 4,
    attentionKEqV: false,
    hiddenActivation: "silu",

    linearAttentionLayerCount: 48,
    linearNumKeyHeads: 16,
    linearKeyHeadDim: 128,
    linearNumValueHeads: 48,
    linearValueHeadDim: 128,
    linearConvKernelDim: 4,

    totalParamsB: 27,
    totalExpertParamsB: 0,
    estimatedWeightsGb: 27
  },
  {
    ...sharedQwen36Definition,
    id: "qwen3.6-35b-a3b",
    displayName: "Qwen3.6-35B-A3B",
    configSource: "docs/Qwen_3.6/config/Qwen3.6-35B-A3B-config.json",
    parameterSourceUrl: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json",
    weightSourceUrl: "https://huggingface.co/Qwen/Qwen3.6-35B-A3B",
    recommendedPrecision: {
      label: "BF16 weights / BF16 experts / BF16 activations",
      bytesPerWeight: 2,
      bytesPerActivation: 2,
      bytesPerExpert: 2
    },
    estimatedWeightsGb: 69.32
  }
];
