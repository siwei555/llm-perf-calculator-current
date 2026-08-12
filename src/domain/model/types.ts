export type ModelFamily = string;

export type ModelId = string;

export type FormulaStrategyId =
  | "deepseek-v4-compressed-moe"
  | "dense-decoder-moe"
  | "dense-decoder-transformer"
  | "hybrid-linear-moe"
  | "hybrid-linear-dense";

export type ModelDefinition = {
  family: ModelFamily;
  id: ModelId;
  displayName: string;
  architectureKind:
    | "compressed-moe"
    | "dense-decoder"
    | "dense-decoder-moe"
    | "hybrid-linear-moe"
    | "hybrid-linear-dense";
  formulaStrategyId: FormulaStrategyId;
  configSource?: string;
  /** 官方 config.json 文件，用于追溯模型参数与配置来源 */
  parameterSourceUrl?: string;
  /** 官方模型仓库，用于追溯权重文件来源 */
  weightSourceUrl?: string;
  /** 选择模型时应用到平台输入的推荐精度；用户仍可在计算页手动覆盖 */
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

  // —— compressed-moe / DeepSeek V4 专用 ——
  qLoraRank: number;
  oLoraRank: number;
  oGroups: number;
  indexHeads: number;
  indexHeadDim: number;
  indexTopk: number;
  csaCompressRate: number;
  hcaCompressRate: number;
  moeExperts: number;
  activeExperts: number;
  moeIntermediateSize: number;
  csaLayerCount: number;
  hcaLayerCount: number;

  // —— 通用 ——
  slidingWindow: number;

  // —— dense-decoder 专用 (optional for compressed-moe) ——
  /** FFN intermediate_size（dense 模型用，替代 moeIntermediateSize） */
  intermediateSize?: number;
  /** 全局注意力（full_attention）层数 */
  fullAttentionLayerCount?: number;
  /** 滑动窗口注意力（sliding_attention）层数 */
  slidingAttentionLayerCount?: number;
  /** full_attention 层的 head_dim */
  globalHeadDim?: number;
  /** full_attention 层的 KV 头数（MQA 通常为 1） */
  numGlobalKeyValueHeads?: number;
  /** full_attention 层 K=V 是否共享投影 */
  attentionKEqV?: boolean;
  /** FFN 激活函数 */
  hiddenActivation?: string;
  /** Complete checkpoint parameter count (B), used for resident weight memory. */
  checkpointParamsB?: number;
  /** Text-backbone parameter count (B), excluding vision/audio encoders. */
  textBackboneParamsB?: number;
  /** Embedding-table parameters (B) accessed by row rather than scanned per token. */
  tokenLookupParamsB?: number;
  /** Gemma Per-Layer Embedding width; zero/undefined means PLE is disabled. */
  perLayerEmbeddingSize?: number;
  /** Number of trailing layers that reuse K/V from earlier layers of the same type. */
  kvSharedLayerCount?: number;
  independentSlidingAttentionLayerCount?: number;
  independentFullAttentionLayerCount?: number;
  /** Whether the KV-sharing region uses 2x intermediate width. */
  doubleWideMlpInKvSharedLayers?: boolean;

  // —— hybrid-linear-moe / hybrid-linear-dense 专用 ——
  /** Gated DeltaNet 线性注意层数 */
  linearAttentionLayerCount?: number;
  /** 线性注意 key 头数 */
  linearNumKeyHeads?: number;
  /** 线性注意 key 头维度 */
  linearKeyHeadDim?: number;
  /** 线性注意 value 头数 */
  linearNumValueHeads?: number;
  /** 线性注意 value 头维度 */
  linearValueHeadDim?: number;
  /** 线性注意 conv kernel 大小 */
  linearConvKernelDim?: number;

  // —— 同时适用 ——
  /** DeepSeek V4: 纯 sliding (无压缩) 层数; Dense: 等价 slidingAttentionLayerCount */
  slidingLayerCount: number;

  /** 总参数量（B = 10^9），用于 weight memory 计算 */
  totalParamsB: number;
  /** 专家参数量（B = 10^9），MoE 模型用；dense 模型填 0 */
  totalExpertParamsB: number;

  /** @deprecated 由 bytesPerWeight / bytesPerExpert 动态计算替代 */
  estimatedWeightsGb: number;
};
