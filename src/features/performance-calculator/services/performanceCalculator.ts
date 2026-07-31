import type { ModelDefinition } from "../../../domain/model/types";
import type {
  BottleneckType,
  ComparisonRow,
  FormulaTraceSection,
  IntermediateMetric,
  PerformanceResult,
  TokenSweepPoint
} from "../../../domain/performance/types";
import type { PlatformInput } from "../../../domain/platform/types";
import type { WorkloadInput } from "../../../domain/workload/types";

type LayerBreakdown = {
  q: number;
  kvProj: number;
  core: number;
  compressor: number;
  indexerLin: number;
  indexerAttn: number;
  output: number;
  moe: number;
  total: number;
};

type FullComputation = {
  prefillFlops: number;
  decodeBytes: number;
  decodeCacheBytes: number;
  decodeWeightBytes: number;
  weightGb: number;
  cacheGb: number;
  persistentSlidingCacheBytes: number;
  persistentHcaCacheBytes: number;
  persistentCsaCacheBytes: number;
  persistentSlidingCacheTotalBytes: number;
  persistentHcaCacheTotalBytes: number;
  persistentCsaCacheTotalBytes: number;
  tmpPeakGb: number;
  tmpPeakBytes: number;
  tmpPeakLkv: number;
  overheadGb: number;
  totalRuntimeMemoryGb: number;
  prefillComputeTps: number;
  prefillBandwidthTps: number;
  decodeComputeTps: number;
  decodeBandwidthTps: number;
  decodeComputeFlopsPerToken: number;
  decodeSlidingLkv: number;
  decodeCsaLkv: number;
  decodeHcaLkv: number;
  decodeSlidingBytesPerToken: number;
  decodeCsaBytesPerToken: number;
  decodeHcaBytesPerToken: number;
  prefillTps: number;
  decodeTps: number;
  ttftMs: number;
  prefillBottleneck: BottleneckType;
  decodeBottleneck: BottleneckType;
  memoryFitsCapacity: boolean;
  slidingLayer: LayerBreakdown;
  csaLayer: LayerBreakdown;
  hcaLayer: LayerBreakdown;
};

function gbpsToBytesPerSecond(valueGbps: number) {
  return valueGbps * 1_000_000_000;
}

function toFlops(valueTflops: number) {
  return valueTflops * 1_000_000_000_000;
}

function bytesToGb(bytes: number) {
  return bytes / 1_000_000_000;
}

/** 根据字节/参数设置动态计算权重显存 */
function computeWeightGb(model: ModelDefinition, platform: PlatformInput): number {
  const nonexpertB = model.totalParamsB - model.totalExpertParamsB;
  const expertB = model.totalExpertParamsB;
  return nonexpertB * platform.bytesPerWeight + expertB * platform.bytesPerExpert;
}

function formatTflops(value: number) {
  return `${(value / 1e12).toFixed(2)} T`;
}

function formatGflops(value: number) {
  return `${(value / 1e9).toFixed(2)} G`;
}

function formatMb(value: number) {
  return `${(value / 1_000_000).toFixed(2)} MB`;
}

function formatLayerContribution(
  layer: LayerBreakdown,
  layerCount: number,
  key: keyof LayerBreakdown
) {
  if (layerCount === 0) {
    return "0.00 T";
  }

  return `${formatTflops(layer[key])} x ${layerCount} = ${formatTflops(layer[key] * layerCount)}`;
}

function inferBottleneck(
  computeLimit: number,
  bandwidthLimit: number
): BottleneckType {
  return computeLimit <= bandwidthLimit ? "compute-bound" : "bandwidth-bound";
}

function flopsQ(model: ModelDefinition, sequenceLength: number) {
  return (
    2 *
    sequenceLength *
    (model.hiddenSize * model.qLoraRank +
      model.qLoraRank * model.attentionHeads * model.headDim)
  );
}

function flopsKvProj(model: ModelDefinition, sequenceLength: number) {
  return 2 * sequenceLength * model.hiddenSize * model.headDim;
}

function flopsCore(
  model: ModelDefinition,
  sequenceLength: number,
  layerType: "sliding" | "csa" | "hca"
) {
  const lkv =
    layerType === "sliding"
      ? model.slidingWindow
      : layerType === "csa"
        ? model.slidingWindow + model.indexTopk
        : model.slidingWindow +
          Math.ceil(sequenceLength / (2 * model.hcaCompressRate));

  return 4 * sequenceLength * lkv * model.attentionHeads * model.headDim;
}

function flopsCompressor(
  model: ModelDefinition,
  sequenceLength: number,
  layerType: "sliding" | "csa" | "hca"
) {
  if (layerType === "sliding") {
    return 0;
  }

  const coeff = layerType === "csa" ? 8 : 4;
  return coeff * sequenceLength * model.hiddenSize * model.headDim;
}

function flopsIndexerLin(model: ModelDefinition, sequenceLength: number) {
  return (
    sequenceLength *
    (8 * model.hiddenSize * model.indexHeadDim +
      2 * model.qLoraRank * model.indexHeads * model.indexHeadDim +
      2 * model.hiddenSize * model.indexHeads)
  );
}

function flopsIndexerAttn(model: ModelDefinition, sequenceLength: number) {
  return (
    sequenceLength *
    sequenceLength *
    model.indexHeads *
    model.indexHeadDim /
    model.csaCompressRate
  );
}

function flopsOutput(model: ModelDefinition, sequenceLength: number) {
  return (
    2 *
    sequenceLength *
    (model.attentionHeads * model.headDim * model.oLoraRank +
      model.oGroups * model.oLoraRank * model.hiddenSize)
  );
}

function flopsMoe(model: ModelDefinition, sequenceLength: number) {
  return (
    6 *
    sequenceLength *
    model.hiddenSize *
    model.moeIntermediateSize *
    (model.activeExperts + 1)
  );
}

function layerBreakdown(
  model: ModelDefinition,
  sequenceLength: number,
  layerType: "sliding" | "csa" | "hca"
): LayerBreakdown {
  const q = flopsQ(model, sequenceLength);
  const kvProj = flopsKvProj(model, sequenceLength);
  const core = flopsCore(model, sequenceLength, layerType);
  const compressor = flopsCompressor(model, sequenceLength, layerType);
  const indexerLin = layerType === "csa" ? flopsIndexerLin(model, sequenceLength) : 0;
  const indexerAttn = layerType === "csa" ? flopsIndexerAttn(model, sequenceLength) : 0;
  const output = flopsOutput(model, sequenceLength);
  const moe = flopsMoe(model, sequenceLength);
  const total = q + kvProj + core + compressor + indexerLin + indexerAttn + output + moe;

  return { q, kvProj, core, compressor, indexerLin, indexerAttn, output, moe, total };
}

function decodeCacheBytes(
  model: ModelDefinition,
  batchSize: number,
  contextLength: number,
  e: number
) {
  return (
    model.slidingLayerCount *
      persistentSlidingCacheBytes(model, batchSize, e) +
    model.hcaLayerCount *
      persistentHcaCacheBytes(model, batchSize, contextLength, e) +
    model.csaLayerCount *
      persistentCsaCacheBytes(model, batchSize, contextLength, e)
  );
}

function persistentSlidingCacheBytes(
  model: ModelDefinition,
  batchSize: number,
  e: number
) {
  return batchSize * e * (model.slidingWindow - 1) * model.headDim;
}

function persistentHcaCacheBytes(
  model: ModelDefinition,
  batchSize: number,
  contextLength: number,
  e: number
) {
  return (
    batchSize *
    e *
    ((model.slidingWindow - 1) * model.headDim +
      Math.floor(contextLength / model.hcaCompressRate) * model.headDim +
      2 * (contextLength % model.hcaCompressRate) * model.headDim)
  );
}

function persistentCsaCacheBytes(
  model: ModelDefinition,
  batchSize: number,
  contextLength: number,
  e: number
) {
  return (
    batchSize *
    e *
    ((model.slidingWindow - 1) * model.headDim +
      Math.floor(contextLength / model.csaCompressRate) *
        (model.headDim + model.indexHeadDim) +
      4 *
        (contextLength % model.csaCompressRate) *
        (model.headDim + model.indexHeadDim) +
      2 * model.csaCompressRate * (model.headDim + model.indexHeadDim))
  );
}

function decodeTmpPeakBytes(
  model: ModelDefinition,
  batchSize: number,
  contextLength: number,
  e: number
) {
  const csaLkv = decodeCsaLkv(model, contextLength);
  const hcaLkv = decodeHcaLkv(model, contextLength);
  const peakLkv = Math.max(csaLkv, hcaLkv, model.slidingWindow);

  return batchSize * e * 2 * model.attentionHeads * peakLkv * model.headDim;
}

function decodeCsaLkv(model: ModelDefinition, contextLength: number) {
  return (
    model.slidingWindow +
    Math.min(model.indexTopk, Math.floor(contextLength / model.csaCompressRate))
  );
}

function decodeHcaLkv(model: ModelDefinition, contextLength: number) {
  return model.slidingWindow + Math.floor(contextLength / model.hcaCompressRate);
}

function decodeLayerBytesPerToken(
  model: ModelDefinition,
  lkv: number,
  batchSize: number,
  e: number
) {
  return batchSize * 2 * model.attentionHeads * lkv * model.headDim * e;
}

function decodeBytesPerToken(
  model: ModelDefinition,
  contextLength: number,
  batchSize: number,
  e: number
) {
  const slidingBytes = decodeLayerBytesPerToken(model, model.slidingWindow, batchSize, e);
  const csaBytes = decodeLayerBytesPerToken(
    model,
    decodeCsaLkv(model, contextLength),
    batchSize,
    e
  );
  const hcaBytes = decodeLayerBytesPerToken(
    model,
    decodeHcaLkv(model, contextLength),
    batchSize,
    e
  );

  return (
    model.slidingLayerCount * slidingBytes +
    model.csaLayerCount * csaBytes +
    model.hcaLayerCount * hcaBytes
  );
}

// ─────────────────────────────────────────────────────────────
// Dense Decoder Transformer helpers (Gemma 4, Llama, etc.)
// ─────────────────────────────────────────────────────────────

function denseFlopsQ(model: ModelDefinition, seqLen: number, headDim: number) {
  return 2 * seqLen * model.hiddenSize * model.attentionHeads * headDim;
}

function denseFlopsKvProj(
  model: ModelDefinition,
  seqLen: number,
  nKv: number,
  headDim: number,
  hasVProj: boolean
) {
  const k = 2 * seqLen * model.hiddenSize * nKv * headDim;
  const v = hasVProj ? k : 0;
  return k + v;
}

function denseFlopsCore(
  model: ModelDefinition,
  seqLen: number,
  lkv: number,
  headDim: number,
  causalFactor: number
) {
  return causalFactor * seqLen * lkv * model.attentionHeads * headDim;
}

function denseFlopsOutput(model: ModelDefinition, seqLen: number, headDim: number) {
  return 2 * seqLen * model.attentionHeads * headDim * model.hiddenSize;
}

function denseFlopsMlp(model: ModelDefinition, seqLen: number) {
  if (model.formulaStrategyId === "dense-decoder-moe") {
    return 6 * seqLen * model.hiddenSize * model.moeIntermediateSize * (model.activeExperts + 1);
  }

  const I = model.intermediateSize ?? model.moeIntermediateSize;
  return 6 * seqLen * model.hiddenSize * I;
}

type DenseLayerBreakdown = {
  q: number;
  kvProj: number;
  core: number;
  output: number;
  mlp: number;
  total: number;
};

function denseLayerBreakdown(
  model: ModelDefinition,
  seqLen: number,
  lkv: number,
  headDim: number,
  nKv: number,
  hasVProj: boolean,
  causalFactor: number
): DenseLayerBreakdown {
  const q = denseFlopsQ(model, seqLen, headDim);
  const kvProj = denseFlopsKvProj(model, seqLen, nKv, headDim, hasVProj);
  const core = denseFlopsCore(model, seqLen, lkv, headDim, causalFactor);
  const output = denseFlopsOutput(model, seqLen, headDim);
  const mlp = denseFlopsMlp(model, seqLen);
  const total = q + kvProj + core + output + mlp;
  return { q, kvProj, core, output, mlp, total };
}

function denseCacheBytesPerLayer(
  model: ModelDefinition,
  batchSize: number,
  lkv: number,
  nKv: number,
  headDim: number,
  e: number
) {
  return batchSize * e * lkv * nKv * headDim * 2;
}

function denseDecodeFlopsPerToken(
  model: ModelDefinition,
  contextLength: number,
  slidingLayers: number,
  fullLayers: number,
  cSliding: number,
  cFull: number,
  nKvSliding: number,
  nKvFull: number,
  hasVProjSliding: boolean,
  hasVProjFull: boolean,
  decodeSlidingLkv: number,
  decodeFullLkv: number
) {
  const D = model.hiddenSize;
  const n_h = model.attentionHeads;
  const I = model.moeIntermediateSize;
  const k = model.activeExperts;

  const slidingQ = 2 * D * n_h * cSliding;
  const slidingKv = 2 * D * nKvSliding * cSliding * (hasVProjSliding ? 2 : 1);
  const slidingCore = 4 * decodeSlidingLkv * n_h * cSliding;
  const slidingOutput = 2 * n_h * cSliding * D;

  const fullQ = 2 * D * n_h * cFull;
  const fullKv = 2 * D * nKvFull * cFull * (hasVProjFull ? 2 : 1);
  const fullCore = 4 * contextLength * n_h * cFull;
  const fullOutput = 2 * n_h * cFull * D;

  const moe = 6 * D * I * (k + 1);

  return (
    slidingLayers * (slidingQ + slidingKv + slidingCore + slidingOutput + moe) +
    fullLayers * (fullQ + fullKv + fullCore + fullOutput + moe)
  );
}

function computeDenseFullResult(
  model: ModelDefinition,
  platform: PlatformInput,
  workload: WorkloadInput,
  sequenceLength: number
): FullComputation {
  const S = sequenceLength;
  const B = platform.batchSize;
  const S_ctx = workload.prefillTokenLength;

  const slidingLayers = model.slidingAttentionLayerCount ?? model.slidingLayerCount;
  const fullLayers = model.fullAttentionLayerCount ?? 0;
  const nWin = model.slidingWindow;

  const cSliding = model.headDim;
  const nKvSliding = model.kvHeads;
  const hasVProjSliding = true;

  const cFull = model.globalHeadDim ?? model.headDim;
  const nKvFull = model.numGlobalKeyValueHeads ?? model.kvHeads;
  const hasVProjFull = !(model.attentionKEqV ?? false);

  const causalFactor = 2;

  // ---- Prefill FLOPs ----
  const slidingBreakdown = denseLayerBreakdown(model, S, nWin, cSliding, nKvSliding, hasVProjSliding, causalFactor);
  const fullBreakdown = denseLayerBreakdown(model, S, S, cFull, nKvFull, hasVProjFull, causalFactor);

  const prefillFlops =
    slidingLayers * slidingBreakdown.total + fullLayers * fullBreakdown.total;

  // ---- Memory ----
  const e = platform.bytesPerActivation;
  const weightGb = computeWeightGb(model, platform);

  const slidingCachePerLayer = denseCacheBytesPerLayer(model, B, nWin, nKvSliding, cSliding, e);
  const fullCachePerLayer = denseCacheBytesPerLayer(model, B, S_ctx, nKvFull, cFull, e);
  const slidingCacheTotal = slidingLayers * slidingCachePerLayer;
  const fullCacheTotal = fullLayers * fullCachePerLayer;
  const cacheBytes = slidingCacheTotal + fullCacheTotal;
  const cacheGb = bytesToGb(cacheBytes);

  const decodeSlidingLkv = nWin + 1;
  const decodeFullLkv = S_ctx + 1;
  const tmpPeakLkv = Math.max(decodeSlidingLkv, decodeFullLkv);
  const tmpPeakBytes = B * 2 * 2 * model.attentionHeads * tmpPeakLkv * Math.max(cSliding, cFull);
  const tmpPeakGb = bytesToGb(tmpPeakBytes);

  const overheadGb = platform.runtimeOverheadGb;
  const totalRuntimeMemoryGb = weightGb + cacheGb + tmpPeakGb + overheadGb;
  const memoryFitsCapacity = totalRuntimeMemoryGb <= platform.memoryCapacityGb;

  // ---- TPS ----
  const effectiveCompute = toFlops(platform.computeThroughputTflops * platform.computeEfficiency);
  const effectiveBandwidth = gbpsToBytesPerSecond(
    platform.memoryBandwidthGbps * platform.bandwidthEfficiency
  );

  const prefillComputeTps = (effectiveCompute * S) / prefillFlops;
  const prefillTrafficBytes = weightGb * 1_000_000_000 + cacheGb * 1_000_000_000 * 0.1;
  const prefillBandwidthTps = (effectiveBandwidth * S) / prefillTrafficBytes;

  const decodeSlidingBytesPerToken = denseCacheBytesPerLayer(model, B, decodeSlidingLkv, nKvSliding, cSliding, e);
  const decodeFullBytesPerToken = denseCacheBytesPerLayer(model, B, decodeFullLkv, nKvFull, cFull, e);
  const decodeCacheTrafficBytes =
    slidingLayers * decodeSlidingBytesPerToken + fullLayers * decodeFullBytesPerToken;
  const isDenseMoe = model.formulaStrategyId === "dense-decoder-moe";
  const activeExpertFraction = model.moeExperts > 0
    ? model.activeExperts / model.moeExperts
    : 1;
  const nonExpertB = model.totalParamsB - model.totalExpertParamsB;
  const expertB = model.totalExpertParamsB;
  const decodeWeightBytes = isDenseMoe
    ? nonExpertB * 1_000_000_000 * platform.bytesPerWeight +
      expertB * 1_000_000_000 * activeExpertFraction * platform.bytesPerExpert
    : weightGb * 1_000_000_000;
  const decodeBytes = decodeWeightBytes + decodeCacheTrafficBytes;

  const decodeComputeFlopsPerToken = isDenseMoe
    ? denseDecodeFlopsPerToken(
      model,
      S_ctx,
      slidingLayers,
      fullLayers,
      cSliding,
      cFull,
      nKvSliding,
      nKvFull,
      hasVProjSliding,
      hasVProjFull,
      decodeSlidingLkv,
      decodeFullLkv
    )
    : 6 * model.hiddenSize * (model.intermediateSize ?? model.moeIntermediateSize);

  const decodeComputeTps = effectiveCompute / decodeComputeFlopsPerToken;
  const decodeBandwidthTps = effectiveBandwidth / decodeBytes;

  const prefillTps = Math.min(prefillComputeTps, prefillBandwidthTps);
  const decodeTps = Math.min(decodeComputeTps, decodeBandwidthTps);
  const ttftMs = (S / Math.max(prefillTps, 1e-6)) * 1000;

  const toLayerBreakdown = (d: DenseLayerBreakdown): LayerBreakdown => ({
    q: d.q,
    kvProj: d.kvProj,
    core: d.core,
    compressor: 0,
    indexerLin: 0,
    indexerAttn: 0,
    output: d.output,
    moe: d.mlp,
    total: d.total
  });

  return {
    prefillFlops,
    decodeBytes,
    decodeCacheBytes: decodeCacheTrafficBytes,
    decodeWeightBytes,
    weightGb,
    cacheGb,
    persistentSlidingCacheBytes: slidingCachePerLayer,
    persistentHcaCacheBytes: 0,
    persistentCsaCacheBytes: fullCachePerLayer,
    persistentSlidingCacheTotalBytes: slidingCacheTotal,
    persistentHcaCacheTotalBytes: 0,
    persistentCsaCacheTotalBytes: fullCacheTotal,
    tmpPeakGb,
    tmpPeakBytes,
    tmpPeakLkv,
    overheadGb,
    totalRuntimeMemoryGb,
    prefillComputeTps,
    prefillBandwidthTps,
    decodeComputeTps,
    decodeBandwidthTps,
    decodeComputeFlopsPerToken,
    decodeSlidingLkv,
    decodeCsaLkv: decodeFullLkv,
    decodeHcaLkv: 0,
    decodeSlidingBytesPerToken,
    decodeCsaBytesPerToken: decodeFullBytesPerToken,
    decodeHcaBytesPerToken: 0,
    prefillTps,
    decodeTps,
    ttftMs,
    prefillBottleneck: inferBottleneck(prefillComputeTps, prefillBandwidthTps),
    decodeBottleneck: inferBottleneck(decodeComputeTps, decodeBandwidthTps),
    memoryFitsCapacity,
    slidingLayer: toLayerBreakdown(slidingBreakdown),
    csaLayer: toLayerBreakdown(fullBreakdown),
    hcaLayer: { q: 0, kvProj: 0, core: 0, compressor: 0, indexerLin: 0, indexerAttn: 0, output: 0, moe: 0, total: 0 }
  };
}

function computeFullResult(
  model: ModelDefinition,
  platform: PlatformInput,
  workload: WorkloadInput,
  sequenceLength: number
): FullComputation {
  const slidingLayer = layerBreakdown(model, sequenceLength, "sliding");
  const csaLayer = layerBreakdown(model, sequenceLength, "csa");
  const hcaLayer = layerBreakdown(model, sequenceLength, "hca");
  const prefillFlops =
    model.slidingLayerCount * slidingLayer.total +
    model.csaLayerCount * csaLayer.total +
    model.hcaLayerCount * hcaLayer.total;
  const e = platform.bytesPerActivation;
  const weightGb = computeWeightGb(model, platform);
  const persistentSlidingCache = persistentSlidingCacheBytes(
    model,
    platform.batchSize,
    e
  );
  const persistentHcaCache = persistentHcaCacheBytes(
    model,
    platform.batchSize,
    workload.prefillTokenLength,
    e
  );
  const persistentCsaCache = persistentCsaCacheBytes(
    model,
    platform.batchSize,
    workload.prefillTokenLength,
    e
  );
  const persistentSlidingCacheTotal =
    model.slidingLayerCount * persistentSlidingCache;
  const persistentHcaCacheTotal = model.hcaLayerCount * persistentHcaCache;
  const persistentCsaCacheTotal = model.csaLayerCount * persistentCsaCache;
  const cacheBytes =
    persistentSlidingCacheTotal + persistentHcaCacheTotal + persistentCsaCacheTotal;
  const cacheGb = bytesToGb(cacheBytes);
  const tmpPeakBytes = decodeTmpPeakBytes(
    model,
    platform.batchSize,
    workload.prefillTokenLength,
    e
  );
  const tmpPeakGb = bytesToGb(tmpPeakBytes);
  const tmpPeakLkv = Math.max(
    decodeCsaLkv(model, workload.prefillTokenLength),
    decodeHcaLkv(model, workload.prefillTokenLength),
    model.slidingWindow
  );
  const overheadGb = platform.runtimeOverheadGb;
  const totalRuntimeMemoryGb = weightGb + cacheGb + tmpPeakGb + overheadGb;
  const memoryFitsCapacity = totalRuntimeMemoryGb <= platform.memoryCapacityGb;
  const effectiveCompute = toFlops(platform.computeThroughputTflops * platform.computeEfficiency);
  const effectiveBandwidth = gbpsToBytesPerSecond(
    platform.memoryBandwidthGbps * platform.bandwidthEfficiency
  );
  const prefillComputeTps = (effectiveCompute * sequenceLength) / prefillFlops;
  const prefillTrafficBytes =
    (weightGb * 1_000_000_000 + cacheGb * 1_000_000_000 * 0.1) * platform.batchSize;
  const prefillBandwidthTps = (effectiveBandwidth * sequenceLength) / prefillTrafficBytes;
  const decodeSlidingLkv = model.slidingWindow;
  const decodeCsaVisibleLength = decodeCsaLkv(model, workload.prefillTokenLength);
  const decodeHcaVisibleLength = decodeHcaLkv(model, workload.prefillTokenLength);
  const decodeSlidingBytesPerToken = decodeLayerBytesPerToken(
    model,
    decodeSlidingLkv,
    platform.batchSize,
    e
  );
  const decodeCsaBytesPerToken = decodeLayerBytesPerToken(
    model,
    decodeCsaVisibleLength,
    platform.batchSize,
    e
  );
  const decodeHcaBytesPerToken = decodeLayerBytesPerToken(
    model,
    decodeHcaVisibleLength,
    platform.batchSize,
    e
  );
  const decodeCacheTrafficBytes = decodeBytesPerToken(
    model,
    workload.prefillTokenLength,
    platform.batchSize,
    e
  );
  // MoE decode: only active expert weights read per token
  const nonExpertB = model.totalParamsB - model.totalExpertParamsB;
  const expertB = model.totalExpertParamsB;
  const activeExpertFraction = model.moeExperts > 0
    ? model.activeExperts / model.moeExperts
    : 1;
  const decodeWeightBytes =
    nonExpertB * 1_000_000_000 * platform.bytesPerWeight +
    expertB * 1_000_000_000 * activeExpertFraction * platform.bytesPerExpert;
  const decodeBytes = decodeWeightBytes + decodeCacheTrafficBytes;
  const decodeComputeFlopsPerToken =
    (model.decoderLayers * model.hiddenSize * model.moeIntermediateSize * (model.activeExperts + 1)) /
    3;
  const decodeComputeTps = effectiveCompute / decodeComputeFlopsPerToken;
  const decodeBandwidthTps = effectiveBandwidth / decodeBytes;
  const prefillTps = Math.min(prefillComputeTps, prefillBandwidthTps);
  const decodeTps = Math.min(decodeComputeTps, decodeBandwidthTps);
  const ttftMs = (sequenceLength / Math.max(prefillTps, 1e-6)) * 1000;

  return {
    prefillFlops,
    decodeBytes,
    decodeCacheBytes: decodeCacheTrafficBytes,
    decodeWeightBytes,
    weightGb,
    cacheGb,
    persistentSlidingCacheBytes: persistentSlidingCache,
    persistentHcaCacheBytes: persistentHcaCache,
    persistentCsaCacheBytes: persistentCsaCache,
    persistentSlidingCacheTotalBytes: persistentSlidingCacheTotal,
    persistentHcaCacheTotalBytes: persistentHcaCacheTotal,
    persistentCsaCacheTotalBytes: persistentCsaCacheTotal,
    tmpPeakGb,
    tmpPeakBytes,
    tmpPeakLkv,
    overheadGb,
    totalRuntimeMemoryGb,
    prefillComputeTps,
    prefillBandwidthTps,
    decodeComputeTps,
    decodeBandwidthTps,
    decodeComputeFlopsPerToken,
    decodeSlidingLkv,
    decodeCsaLkv: decodeCsaVisibleLength,
    decodeHcaLkv: decodeHcaVisibleLength,
    decodeSlidingBytesPerToken,
    decodeCsaBytesPerToken,
    decodeHcaBytesPerToken,
    prefillTps,
    decodeTps,
    ttftMs,
    prefillBottleneck: inferBottleneck(prefillComputeTps, prefillBandwidthTps),
    decodeBottleneck: inferBottleneck(decodeComputeTps, decodeBandwidthTps),
    memoryFitsCapacity,
    slidingLayer,
    csaLayer,
    hcaLayer
  };
}

function buildDenseFormulaTrace(
  model: ModelDefinition,
  workload: WorkloadInput,
  result: FullComputation
): FormulaTraceSection[] {
  const isDenseMoe = model.formulaStrategyId === "dense-decoder-moe";
  const slidingLayers = model.slidingAttentionLayerCount ?? model.slidingLayerCount;
  const fullLayers = model.fullAttentionLayerCount ?? 0;
  const cFull = model.globalHeadDim ?? model.headDim;
  const nKvFull = model.numGlobalKeyValueHeads ?? model.kvHeads;

  const slidingTotal = result.slidingLayer.total * slidingLayers;
  const fullTotal = result.csaLayer.total * fullLayers;

  return [
    {
      category: "prefill",
      rows: [
        {
          label: "Sliding layer Q path FLOPs",
          expression: "2 · S · D · n_h · c_s",
          evaluated: `${formatTflops(result.slidingLayer.q)} per layer × ${slidingLayers} = ${formatTflops(result.slidingLayer.q * slidingLayers)}`
        },
        {
          label: "Full layer Q path FLOPs",
          expression: "2 · S · D · n_h · c_f",
          evaluated: `${formatTflops(result.csaLayer.q)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.q * fullLayers)}`
        },
        {
          label: "Sliding layer KV proj FLOPs",
          expression: "2 · S · D · n_kv_s · c_s · 2",
          evaluated: `${formatTflops(result.slidingLayer.kvProj)} per layer × ${slidingLayers} = ${formatTflops(result.slidingLayer.kvProj * slidingLayers)}`
        },
        {
          label: "Full layer KV proj FLOPs",
          expression: `2 · S · D · n_kv_f(${nKvFull}) · c_f(${cFull}) · (1 + has_v_proj)`,
          evaluated: `${formatTflops(result.csaLayer.kvProj)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.kvProj * fullLayers)}`
        },
        {
          label: "Sliding core attention FLOPs",
          expression: "2 · S · n_win · n_h · c_s  (FA/SDPA causal)",
          evaluated: `${formatTflops(result.slidingLayer.core)} per layer × ${slidingLayers} = ${formatTflops(result.slidingLayer.core * slidingLayers)}`
        },
        {
          label: "Full core attention FLOPs (FA/SDPA)",
          expression: "2 · S² · n_h · c_f  (causal ≈ S²/2 pairs)",
          evaluated: `${formatTflops(result.csaLayer.core)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.core * fullLayers)}`
        },
        {
          label: "Sliding output path FLOPs",
          expression: "2 · S · n_h · c_s · D",
          evaluated: `${formatTflops(result.slidingLayer.output)} per layer × ${slidingLayers} = ${formatTflops(result.slidingLayer.output * slidingLayers)}`
        },
        {
          label: "Full output path FLOPs",
          expression: "2 · S · n_h · c_f · D",
          evaluated: `${formatTflops(result.csaLayer.output)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.output * fullLayers)}`
        },
        {
          label: isDenseMoe
            ? "MoE FLOPs (top-k routed experts, all layers)"
            : "MLP FLOPs (GeGLU, all layers)",
          expression: isDenseMoe ? "6 · S · D · I_moe · (k + 1)" : "6 · S · D · I",
          evaluated: `${formatTflops(result.slidingLayer.moe)} per layer × ${model.decoderLayers} = ${formatTflops(result.slidingLayer.moe * model.decoderLayers)}`
        },
        {
          label: "Sliding layer total",
          expression: "F_Q + F_KV + F_core + F_O + F_MLP",
          evaluated: `${formatTflops(result.slidingLayer.total)} per layer × ${slidingLayers} = ${formatTflops(slidingTotal)}`
        },
        {
          label: "Full layer total",
          expression: "F_Q + F_KV + F_core + F_O + F_MLP",
          evaluated: `${formatTflops(result.csaLayer.total)} per layer × ${fullLayers} = ${formatTflops(fullTotal)}`
        },
        {
          label: "Prefill Total FLOPs",
          expression: "L_sliding · F_sliding + L_full · F_full",
          evaluated: `${formatTflops(result.prefillFlops)}`
        }
      ]
    },
    {
      category: "decode",
      rows: [
        {
          label: "Decode sliding attention visible length",
          expression: "L_kv_decode(sliding) = n_win + 1",
          evaluated: `${result.decodeSlidingLkv} tokens`
        },
        {
          label: "Decode full attention visible length",
          expression: "L_kv_decode(full) = S_ctx + 1",
          evaluated: `${result.decodeCsaLkv} tokens`
        },
        {
          label: "Sliding attn bytes per token",
          expression: "B · e · L_kv_decode(sliding) · n_kv_s · c_s · 2 (K+V read)",
          evaluated: `${formatMb(result.decodeSlidingBytesPerToken)} × ${slidingLayers} layers = ${formatMb(
            result.decodeSlidingBytesPerToken * slidingLayers
          )}`
        },
        {
          label: "Full attn bytes per token",
          expression: "B · e · L_kv_decode(full) · n_kv_f · c_f · 2 (K+V read)",
          evaluated: `${formatMb(result.decodeCsaBytesPerToken)} × ${fullLayers} layers = ${formatMb(
            result.decodeCsaBytesPerToken * fullLayers
          )}`
        },
        {
          label: "Decode cache bytes per token",
          expression: "B_cache = L_sliding · B_sliding + L_full · B_full",
          evaluated: formatMb(result.decodeCacheBytes)
        },
        {
          label: "Decode weight bytes per token",
          expression: isDenseMoe
            ? "B_weights = N_non · bpw + N_exp · (k / E) · bpe"
            : "B_weights = M_weights · 10^9",
          evaluated: formatMb(result.decodeWeightBytes)
        },
        {
          label: "Total decode bytes per token",
          expression: "B_decode = B_weights + B_cache",
          evaluated: `${formatMb(result.decodeWeightBytes)} + ${formatMb(
            result.decodeCacheBytes
          )} = ${formatMb(result.decodeBytes)}`
        },
        {
          label: isDenseMoe
            ? "Decode compute FLOPs per token (Sliding+Full+MoE)"
            : "Decode compute FLOPs per token (Dense MLP estimate)",
          expression: isDenseMoe
            ? "L_sliding · F_sliding + L_full · F_full"
            : "6 · D · I",
          evaluated: formatGflops(result.decodeComputeFlopsPerToken)
        },
        {
          label: "Decode compute ceiling",
          expression: "decode_compute_ceiling = effective_compute / FLOPs_per_token",
          evaluated: `${result.decodeComputeTps.toFixed(2)} tokens/s`
        },
        {
          label: "Decode bandwidth ceiling",
          expression: "decode_bandwidth_ceiling = effective_bandwidth / B_decode",
          evaluated: `${result.decodeBandwidthTps.toFixed(2)} tokens/s`
        },
        {
          label: "Effective Decode TPS",
          expression: "min(decode_compute_ceiling, decode_bandwidth_ceiling)",
          evaluated: `${result.decodeTps.toFixed(2)} tokens/s`
        }
      ]
    },
    {
      category: "memory",
      rows: [
        {
          label: "Weight memory",
          expression: "M_weights = N_non · bpw + N_exp · bpe",
          evaluated: `${result.weightGb.toFixed(2)} GB`,
          sourceLabel: "权重文件出处",
          sourceUrl: model.weightSourceUrl
        },
        {
          label: "Persistent sliding cache (per layer)",
          expression: "B · e · n_win · n_kv_s · c_s · 2 (K+V)",
          evaluated: `${formatMb(result.persistentSlidingCacheBytes)} × ${slidingLayers} layers = ${formatMb(result.persistentSlidingCacheTotalBytes)}`
        },
        {
          label: "Persistent full cache (per layer)",
          expression: "B · e · S_ctx · n_kv_f · c_f · 2 (K+V, torch.cat separate)",
          evaluated: `${formatMb(result.persistentCsaCacheBytes)} × ${fullLayers} layers = ${formatMb(result.persistentCsaCacheTotalBytes)}`
        },
        {
          label: "Persistent Decode Cache",
          expression: "M_cache = M_sliding + M_full",
          evaluated: `${result.cacheGb.toFixed(3)} GB`
        },
        {
          label: "Single-step Temp Peak",
          expression: "B · e · 2 · n_h · max(L_kv_decode) · max(c_s, c_f)",
          evaluated: `${formatMb(result.tmpPeakBytes)}; max L_kv = ${result.tmpPeakLkv}`
        },
        {
          label: "Runtime overhead",
        expression: "M_overhead = user runtime-overhead assumption",
          evaluated: `${result.overheadGb.toFixed(2)} GB`
        },
        {
          label: "Decode base memory",
          expression: "M_weights + M_cache",
          evaluated: `${(result.weightGb + result.cacheGb).toFixed(2)} GB`
        },
        {
          label: "Decode peak memory",
          expression: "M_weights + M_cache + M_tmp + M_overhead",
          evaluated: `${result.totalRuntimeMemoryGb.toFixed(2)} GB`
        }
      ]
    }
  ];
}

function buildComparisonRows(result: FullComputation): ComparisonRow[] {
  return [
    {
      label: "Dominant Cost",
      unit: "-",
      prefill: result.prefillBottleneck,
      decode: result.decodeBottleneck
    },
    {
      label: "Compute Ceiling",
      unit: "tokens/s",
      prefill: result.prefillComputeTps.toFixed(2),
      decode: result.decodeComputeTps.toFixed(2)
    },
    {
      label: "Bandwidth Ceiling",
      unit: "tokens/s",
      prefill: result.prefillBandwidthTps.toFixed(2),
      decode: result.decodeBandwidthTps.toFixed(2)
    },
    {
      label: "Effective Throughput",
      unit: "tokens/s",
      prefill: result.prefillTps.toFixed(2),
      decode: result.decodeTps.toFixed(2)
    },
    {
      label: "Latency",
      unit: "ms",
      prefill: result.ttftMs.toFixed(0),
      decode: (1000 / Math.max(result.decodeTps, 1e-6)).toFixed(2)
    }
  ];
}

function buildIntermediateMetrics(
  model: ModelDefinition,
  platform: PlatformInput,
  workload: WorkloadInput,
  result: FullComputation
): IntermediateMetric[] {
  return [
    {
      key: "hidden_size",
      label: "Hidden Size",
      symbol: "D",
      value: String(model.hiddenSize),
      unit: "-",
      source: "config"
    },
    {
      key: "sequence_length",
      label: "Prefill Sequence Length",
      symbol: "S",
      value: workload.prefillTokenLength.toLocaleString(),
      unit: "tokens",
      source: "config"
    },
    {
      key: "prefill_flops",
      label: "Prefill FLOPs",
      symbol: "F_prefill",
      value: formatTflops(result.prefillFlops),
      unit: "FLOPs",
      source: "formula"
    },
    {
      key: "prefill_compute_ceiling",
      label: "Prefill Compute Ceiling",
      symbol: "TPS_compute",
      value: result.prefillComputeTps.toFixed(2),
      unit: "tokens/s",
      source: "derived"
    },
    {
      key: "decode_bandwidth_bytes",
      label: "Decode Total Traffic Per Token",
      symbol: "B_decode",
      value: (result.decodeBytes / 1_000_000).toFixed(2),
      unit: "MB/token",
      source: "formula"
    },
    {
      key: "decode_weight_bytes",
      label: "Decode Weight Read Per Token",
      symbol: "B_weights",
      value: (result.decodeWeightBytes / 1_000_000).toFixed(2),
      unit: "MB/token",
      source: "formula"
    },
    {
      key: "decode_cache_bytes",
      label: "Decode Cache Traffic Per Token",
      symbol: "B_cache",
      value: (result.decodeCacheBytes / 1_000_000).toFixed(2),
      unit: "MB/token",
      source: "formula"
    },
    {
      key: "weights_memory",
      label: "Weight Memory",
      symbol: "M_weights",
      value: result.weightGb.toFixed(2),
      unit: "GB",
      source: "config"
    },
    {
      key: "decode_cache",
      label: "Persistent Decode Cache",
      symbol: "M_cache",
      value: result.cacheGb.toFixed(3),
      unit: "GB",
      source: "formula"
    },
    {
      key: "tmp_peak",
      label: "Single-step Temp Peak",
      symbol: "M_tmp",
      value: result.tmpPeakGb.toFixed(3),
      unit: "GB",
      source: "formula"
    },
    {
      key: "memory_capacity",
      label: "Capacity",
      symbol: "M_cap",
      value: platform.memoryCapacityGb.toFixed(0),
      unit: "GB",
      source: "config"
    }
  ];
}

function buildFormulaTrace(
  model: ModelDefinition,
  workload: WorkloadInput,
  result: FullComputation
): FormulaTraceSection[] {
  if (
    model.formulaStrategyId === "dense-decoder-transformer" ||
    model.formulaStrategyId === "dense-decoder-moe"
  ) {
    return buildDenseFormulaTrace(model, workload, result);
  }

  if (
    model.formulaStrategyId === "hybrid-linear-moe" ||
    model.formulaStrategyId === "hybrid-linear-dense"
  ) {
    return buildHybridFormulaTrace(model, workload, result);
  }

  return [
    {
      category: "prefill",
      rows: [
        {
          label: "Q path FLOPs",
          expression: "2 * S * (D * r_q + r_q * n_h * c)",
          evaluated: `${formatTflops(result.csaLayer.q)} per layer; all layers = ${formatTflops(
            result.csaLayer.q * model.decoderLayers
          )}`
        },
        {
          label: "KV projection FLOPs",
          expression: "2 * S * D * c",
          evaluated: `${formatTflops(result.csaLayer.kvProj)} per layer; all layers = ${formatTflops(
            result.csaLayer.kvProj * model.decoderLayers
          )}`
        },
        {
          label: "Sliding core attention FLOPs",
          expression: "4 * S * sliding_window * n_h * c",
          evaluated: formatLayerContribution(
            result.slidingLayer,
            model.slidingLayerCount,
            "core"
          )
        },
        {
          label: "CSA core attention FLOPs",
          expression: "4 * S * (sliding_window + index_topk) * n_h * c",
          evaluated: formatLayerContribution(result.csaLayer, model.csaLayerCount, "core")
        },
        {
          label: "HCA core attention FLOPs",
          expression: "4 * S * (sliding_window + ceil(S / (2 * m_hca))) * n_h * c",
          evaluated: formatLayerContribution(result.hcaLayer, model.hcaLayerCount, "core")
        },
        {
          label: "CSA compressor FLOPs",
          expression: "8 * S * D * c",
          evaluated: formatLayerContribution(
            result.csaLayer,
            model.csaLayerCount,
            "compressor"
          )
        },
        {
          label: "HCA compressor FLOPs",
          expression: "4 * S * D * c",
          evaluated: formatLayerContribution(
            result.hcaLayer,
            model.hcaLayerCount,
            "compressor"
          )
        },
        {
          label: "CSA indexer linear FLOPs",
          expression: "S * (8*D*c_I + 2*r_q*n_h_I*c_I + 2*D*n_h_I)",
          evaluated: formatLayerContribution(
            result.csaLayer,
            model.csaLayerCount,
            "indexerLin"
          )
        },
        {
          label: "CSA indexer attn FLOPs",
          expression: "S^2 * n_h^I * c^I / m_csa",
          evaluated: formatLayerContribution(
            result.csaLayer,
            model.csaLayerCount,
            "indexerAttn"
          )
        },
        {
          label: "Output path FLOPs",
          expression: "2 * S * (n_h*c*r_o + o_groups*r_o*D)",
          evaluated: `${formatTflops(result.csaLayer.output)} per layer; all layers = ${formatTflops(
            result.csaLayer.output * model.decoderLayers
          )}`
        },
        {
          label: "MoE FLOPs",
          expression: "6 * S * D * I * (k + 1)",
          evaluated: `${formatTflops(result.csaLayer.moe)} per layer; all layers = ${formatTflops(
            result.csaLayer.moe * model.decoderLayers
          )}`
        },
        {
          label: "Sliding layer total",
          expression: "F_Q + F_KV + F_core + F_O + F_MoE",
          evaluated: formatLayerContribution(
            result.slidingLayer,
            model.slidingLayerCount,
            "total"
          )
        },
        {
          label: "CSA layer total",
          expression: "F_Q + F_KV + F_core + F_compressor + F_indexer + F_O + F_MoE",
          evaluated: formatLayerContribution(result.csaLayer, model.csaLayerCount, "total")
        },
        {
          label: "HCA layer total",
          expression: "F_Q + F_KV + F_core + F_compressor + F_O + F_MoE",
          evaluated: formatLayerContribution(result.hcaLayer, model.hcaLayerCount, "total")
        },
        {
          label: "Total Prefill FLOPs",
          expression: "N_sliding*F_sliding + N_csa*F_csa + N_hca*F_hca",
          evaluated: formatTflops(result.prefillFlops)
        }
      ]
    },
    {
      category: "decode",
      rows: [
        {
          label: "Decode sliding visible length",
          expression: "L_kv_decode(sliding) = sliding_window",
          evaluated: `${result.decodeSlidingLkv} tokens`
        },
        {
          label: "Decode CSA visible length",
          expression: "L_kv_decode(CSA) = sliding_window + min(index_topk, floor(S_ctx / m_csa))",
          evaluated: `${model.slidingWindow} + min(${model.indexTopk}, floor(${workload.prefillTokenLength} / ${model.csaCompressRate})) = ${result.decodeCsaLkv} tokens`
        },
        {
          label: "Decode HCA visible length",
          expression: "L_kv_decode(HCA) = sliding_window + floor(S_ctx / m_hca)",
          evaluated: `${model.slidingWindow} + floor(${workload.prefillTokenLength} / ${model.hcaCompressRate}) = ${result.decodeHcaLkv} tokens`
        },
        {
          label: "Sliding bytes per token",
          expression: "B * 2 * n_h * L_kv * c * bytes_per_elem",
          evaluated: `${formatMb(result.decodeSlidingBytesPerToken)} x ${model.slidingLayerCount} layers = ${formatMb(
            result.decodeSlidingBytesPerToken * model.slidingLayerCount
          )}`
        },
        {
          label: "CSA bytes per token",
          expression: "B * 2 * n_h * L_kv(CSA) * c * bytes_per_elem",
          evaluated: `${formatMb(result.decodeCsaBytesPerToken)} x ${model.csaLayerCount} layers = ${formatMb(
            result.decodeCsaBytesPerToken * model.csaLayerCount
          )}`
        },
        {
          label: "HCA bytes per token",
          expression: "B * 2 * n_h * L_kv(HCA) * c * bytes_per_elem",
          evaluated: `${formatMb(result.decodeHcaBytesPerToken)} x ${model.hcaLayerCount} layers = ${formatMb(
            result.decodeHcaBytesPerToken * model.hcaLayerCount
          )}`
        },
        {
          label: "Decode cache bytes per token",
          expression: "N_sliding*B_sliding + N_csa*B_csa + N_hca*B_hca",
          evaluated: formatMb(result.decodeCacheBytes)
        },
        {
          label: "Decode weight bytes per token",
          expression: "B_weights ~= M_weights",
          evaluated: formatMb(result.decodeWeightBytes)
        },
        {
          label: "Total decode bytes per token",
          expression: "B_decode = B_weights + B_cache",
          evaluated: `${formatMb(result.decodeWeightBytes)} + ${formatMb(
            result.decodeCacheBytes
          )} = ${formatMb(result.decodeBytes)}`
        },
        {
          label: "Decode compute FLOPs per token",
          expression: "L * D * I * (k + 1) / 3",
          evaluated: formatGflops(result.decodeComputeFlopsPerToken)
        },
        {
          label: "Decode compute ceiling",
          expression: "effective_compute / FLOPs_per_token",
          evaluated: `${result.decodeComputeTps.toFixed(2)} tokens/s`
        },
        {
          label: "Decode bandwidth ceiling",
          expression: "effective_bandwidth / bytes_per_token",
          evaluated: `${result.decodeBandwidthTps.toFixed(2)} tokens/s`
        },
        {
          label: "Raw Decode TPS",
          expression: "min(decode_compute_ceiling, decode_bandwidth_ceiling)",
          evaluated: `${Math.min(result.decodeComputeTps, result.decodeBandwidthTps).toFixed(
            2
          )} tokens/s`
        },
        {
          label: "Effective Decode TPS",
          expression: "min(decode_compute_ceiling, decode_bandwidth_ceiling)",
          evaluated: `${result.decodeTps.toFixed(2)} tokens/s`
        }
      ]
    },
    {
      category: "memory",
      rows: [
        {
          label: "Weight memory",
          expression: "M_weights ~= quantized resident model weights",
          evaluated: `${result.weightGb.toFixed(2)} GB`,
          sourceLabel: "权重文件出处",
          sourceUrl: model.weightSourceUrl
        },
        {
          label: "Persistent sliding cache",
          expression: "B * e * (sliding_window - 1) * c",
          evaluated: `${formatMb(result.persistentSlidingCacheBytes)} x ${model.slidingLayerCount} layers = ${formatMb(
            result.persistentSlidingCacheTotalBytes
          )}`
        },
        {
          label: "Persistent HCA cache",
          expression: "B * e * [(n_win-1)*c + floor(S_ctx/m_hca)*c + 2*(S_ctx mod m_hca)*c]",
          evaluated: `${formatMb(result.persistentHcaCacheBytes)} x ${model.hcaLayerCount} layers = ${formatMb(
            result.persistentHcaCacheTotalBytes
          )}`
        },
        {
          label: "Persistent CSA cache",
          expression: "B * e * [(n_win-1)*c + floor(S_ctx/m_csa)*(c+c_I) + 4*(S_ctx mod m_csa)*(c+c_I) + 2*m_csa*(c+c_I)]",
          evaluated: `${formatMb(result.persistentCsaCacheBytes)} x ${model.csaLayerCount} layers = ${formatMb(
            result.persistentCsaCacheTotalBytes
          )}`
        },
        {
          label: "Persistent Decode Cache",
          expression: "M_decode_cache = M_sliding + M_hca + M_csa",
          evaluated: `${result.cacheGb.toFixed(3)} GB`
        },
        {
          label: "Single-step Temp Peak",
          expression: "M_tmp_peak = B * e * 2 * n_h * max(L_kv_decode) * c",
          evaluated: `${formatMb(result.tmpPeakBytes)}; max L_kv = ${result.tmpPeakLkv}`
        },
        {
          label: "Runtime overhead",
        expression: "M_runtime_overhead = user runtime-overhead assumption",
          evaluated: `${result.overheadGb.toFixed(2)} GB`
        },
        {
          label: "Decode base memory",
          expression: "M_weights + M_decode_cache",
          evaluated: `${(result.weightGb + result.cacheGb).toFixed(2)} GB`
        },
        {
          label: "Decode peak memory",
          expression: "M_weights + M_decode_cache + M_tmp_peak + M_runtime_overhead",
          evaluated: `${result.totalRuntimeMemoryGb.toFixed(2)} GB`
        }
      ]
    }
  ];
}

// ─────────────────────────────────────────────────────────────
// Hybrid Linear-MoE helpers (Qwen3.5 Gated DeltaNet + Full GQA)
// ─────────────────────────────────────────────────────────────

function computeHybridLinearMoeResult(
  model: ModelDefinition,
  platform: PlatformInput,
  workload: WorkloadInput,
  sequenceLength: number
): FullComputation {
  const S = sequenceLength;
  const B = platform.batchSize;
  const S_ctx = workload.prefillTokenLength;
  const e = platform.bytesPerActivation;

  const D = model.hiddenSize;
  const n_h = model.attentionHeads;
  const n_kv = model.kvHeads;
  const c = model.headDim;
  const fullLayers = model.fullAttentionLayerCount ?? 0;

  const linearLayers = model.linearAttentionLayerCount ?? 0;
  const n_khL = model.linearNumKeyHeads ?? 0;
  const c_kL = model.linearKeyHeadDim ?? 0;
  const n_vhL = model.linearNumValueHeads ?? 0;
  const c_vL = model.linearValueHeadDim ?? 0;
  const convK = model.linearConvKernelDim ?? 0;
  const keyDim = n_khL * c_kL;
  const valueDim = n_vhL * c_vL;
  const convDim = 2 * keyDim + valueDim;

  const isDenseHybrid = model.formulaStrategyId === "hybrid-linear-dense";
  const I = isDenseHybrid
    ? (model.intermediateSize ?? model.moeIntermediateSize)
    : model.moeIntermediateSize;
  const k = isDenseHybrid ? 0 : model.activeExperts;

  // ── Full Attention layer (GQA + causal) ──
  const fullQGate = 2 * S * D * (2 * n_h * c);  // q_proj: Q+gate combined
  const fullK = 2 * S * D * n_kv * c;
  const fullV = 2 * S * D * n_kv * c;
  const fullCore = 2 * S * S * n_h * c;          // causal ≈ S²/2 pairs × 4 = 2·S²·n_h·c
  const fullOutput = 2 * S * n_h * c * D;
  const fullMoe = 6 * S * D * I * (k + 1);

  const fullLayer: LayerBreakdown = {
    q: fullQGate,
    kvProj: fullK + fullV,
    core: fullCore,
    compressor: 0,
    indexerLin: 0,
    indexerAttn: 0,
    output: fullOutput,
    moe: fullMoe,
    total: fullQGate + fullK + fullV + fullCore + fullOutput + fullMoe
  };

  // ── Linear Attention layer (Gated DeltaNet) ──
  const linQKV = 2 * S * D * convDim;             // in_proj_qkv: D → 2·keyDim+valueDim
  const linZ = 2 * S * D * valueDim;              // in_proj_z: D → valueDim
  const linAB = 2 * S * D * (2 * n_vhL);          // in_proj_a + in_proj_b
  const linConv = 2 * convK * S * convDim;        // Conv1D depthwise
  const linScan = 2 * S * n_vhL * c_kL * c_vL;    // gated delta rule scan
  const linOutput = 2 * S * valueDim * D;         // out_proj: valueDim → D
  const linMoe = fullMoe;                          // same MoE config

  const linearLayer: LayerBreakdown = {
    q: linQKV + linZ,                              // all input projections
    kvProj: linAB,
    core: linScan,
    compressor: linConv,
    indexerLin: 0,
    indexerAttn: 0,
    output: linOutput,
    moe: linMoe,
    total: linQKV + linZ + linAB + linConv + linScan + linOutput + linMoe
  };

  // ── Prefill total ──
  const prefillFlops = fullLayers * fullLayer.total + linearLayers * linearLayer.total;

  // ── Weight memory ──
  const weightGb = computeWeightGb(model, platform);

  // ── Persistent cache ──
  // Full attention: KV cache (K+V, per layer)
  const fullKVPerLayer = B * 2 * n_kv * S_ctx * c * e;
  const fullKVTotal = fullLayers * fullKVPerLayer;

  // Linear attention: conv_state + recurrent_state
  const convStatePerLayer = B * convDim * convK * e;
  // recurrent state: [B, n_v_heads, key_hd, value_hd], fp32 uses 2× bytesPerActivation
  const recurrentStatePerLayer = B * n_vhL * c_kL * c_vL * (e * 2);
  const linearStatePerLayer = convStatePerLayer + recurrentStatePerLayer;
  const linearStateTotal = linearLayers * linearStatePerLayer;

  const cacheBytes = fullKVTotal + linearStateTotal;
  const cacheGb = bytesToGb(cacheBytes);

  // ── Temp peak (full attention repeat_kv) ──
  const fullTmpLkv = S_ctx + 1;
  const tmpPeakBytes = B * 2 * 2 * n_h * fullTmpLkv * c;
  const tmpPeakGb = bytesToGb(tmpPeakBytes);
  const tmpPeakLkv = fullTmpLkv;

  const overheadGb = platform.runtimeOverheadGb;
  const totalRuntimeMemoryGb = weightGb + cacheGb + tmpPeakGb + overheadGb;
  const memoryFitsCapacity = totalRuntimeMemoryGb <= platform.memoryCapacityGb;

  // ── TPS ──
  const effectiveCompute = toFlops(platform.computeThroughputTflops * platform.computeEfficiency);
  const effectiveBandwidth = gbpsToBytesPerSecond(
    platform.memoryBandwidthGbps * platform.bandwidthEfficiency
  );

  const prefillComputeTps = (effectiveCompute * S) / prefillFlops;
  const prefillTrafficBytes =
    (weightGb * 1_000_000_000 + cacheGb * 1_000_000_000 * 0.1) * B;
  const prefillBandwidthTps = (effectiveBandwidth * S) / prefillTrafficBytes;

  // ── Decode FLOPs (single token) ──
  const decodeFullQGate = 2 * D * (2 * n_h * c);
  const decodeFullKV = 2 * D * n_kv * c * 2;           // K + V
  const decodeFullCore = 2 * S_ctx * n_h * c * 2;       // QK^T + AV
  const decodeFullOutput = 2 * n_h * c * D;
  const decodeFullMoePerLayer = 6 * D * I * (k + 1);
  const decodeFullPerLayer =
    decodeFullQGate + decodeFullKV + decodeFullCore + decodeFullOutput + decodeFullMoePerLayer;

  const decodeLinQKV = 2 * D * convDim;
  const decodeLinZ = 2 * D * valueDim;
  const decodeLinAB = 2 * D * (2 * n_vhL);
  const decodeLinConv = 2 * convK * convDim;
  const decodeLinScan = 2 * n_vhL * c_kL * c_vL;
  const decodeLinOutput = 2 * valueDim * D;
  const decodeLinMoePerLayer = decodeFullMoePerLayer;
  const decodeLinPerLayer =
    decodeLinQKV + decodeLinZ + decodeLinAB + decodeLinConv + decodeLinScan + decodeLinOutput + decodeLinMoePerLayer;

  const decodeComputeFlopsPerToken =
    fullLayers * decodeFullPerLayer + linearLayers * decodeLinPerLayer;

  const decodeComputeTps = effectiveCompute / decodeComputeFlopsPerToken;

  // ── Decode traffic bytes per token ──
  const decodeFullBytesPerLayer = B * 2 * n_kv * S_ctx * c * e;    // read K+V
  const decodeLinBytesPerLayer = B * (convDim * convK + n_vhL * c_kL * c_vL) * e;
  const decodeCacheTrafficBytes =
    fullLayers * decodeFullBytesPerLayer + linearLayers * decodeLinBytesPerLayer;

  // MoE decode: only active expert weights read per token
  const nonExpertB = model.totalParamsB - model.totalExpertParamsB;
  const expertB = model.totalExpertParamsB;
  const activeExpertFraction = model.moeExperts > 0
    ? model.activeExperts / model.moeExperts
    : 1;
  const activeDecodeWeightBytes =
    nonExpertB * 1_000_000_000 * platform.bytesPerWeight +
    expertB * 1_000_000_000 * activeExpertFraction * platform.bytesPerExpert;

  const decodeBytes = activeDecodeWeightBytes + decodeCacheTrafficBytes;
  const decodeBandwidthTps = effectiveBandwidth / decodeBytes;

  const prefillTps = Math.min(prefillComputeTps, prefillBandwidthTps);
  const decodeTps = Math.min(decodeComputeTps, decodeBandwidthTps);
  const ttftMs = (S / Math.max(prefillTps, 1e-6)) * 1000;

  return {
    prefillFlops,
    decodeBytes,
    decodeCacheBytes: decodeCacheTrafficBytes,
    decodeWeightBytes: activeDecodeWeightBytes,
    weightGb,
    cacheGb,
    persistentSlidingCacheBytes: 0,
    persistentHcaCacheBytes: linearStatePerLayer,
    persistentCsaCacheBytes: fullKVPerLayer,
    persistentSlidingCacheTotalBytes: 0,
    persistentHcaCacheTotalBytes: linearStateTotal,
    persistentCsaCacheTotalBytes: fullKVTotal,
    tmpPeakGb,
    tmpPeakBytes,
    tmpPeakLkv,
    overheadGb,
    totalRuntimeMemoryGb,
    prefillComputeTps,
    prefillBandwidthTps,
    decodeComputeTps,
    decodeBandwidthTps,
    decodeComputeFlopsPerToken,
    decodeSlidingLkv: 0,
    decodeCsaLkv: S_ctx,
    decodeHcaLkv: 0,
    decodeSlidingBytesPerToken: 0,
    decodeCsaBytesPerToken: decodeFullBytesPerLayer,
    decodeHcaBytesPerToken: decodeLinBytesPerLayer,
    prefillTps,
    decodeTps,
    ttftMs,
    prefillBottleneck: inferBottleneck(prefillComputeTps, prefillBandwidthTps),
    decodeBottleneck: inferBottleneck(decodeComputeTps, decodeBandwidthTps),
    memoryFitsCapacity,
    slidingLayer: { q: 0, kvProj: 0, core: 0, compressor: 0, indexerLin: 0, indexerAttn: 0, output: 0, moe: 0, total: 0 },
    csaLayer: fullLayer,
    hcaLayer: linearLayer
  };
}

function buildHybridFormulaTrace(
  model: ModelDefinition,
  workload: WorkloadInput,
  result: FullComputation
): FormulaTraceSection[] {
  const fullLayers = model.fullAttentionLayerCount ?? 0;
  const linearLayers = model.linearAttentionLayerCount ?? 0;
  const n_vhL = model.linearNumValueHeads ?? 0;
  const c_kL = model.linearKeyHeadDim ?? 0;
  const c_vL = model.linearValueHeadDim ?? 0;
  const isDenseHybrid = model.formulaStrategyId === "hybrid-linear-dense";
  const ffnLabel = isDenseHybrid ? "Dense SwiGLU FFN FLOPs" : "MoE FLOPs (all layers, SwiGLU)";
  const ffnExpression = isDenseHybrid ? "6 · S · D · I" : "6 · S · D · I · (k + 1)";

  const fullTotal = result.csaLayer.total * fullLayers;
  const linearTotal = result.hcaLayer.total * linearLayers;

  return [
    {
      category: "prefill",
      rows: [
        {
          label: "Full layer Q+gate proj FLOPs",
          expression: "2 · S · D · (2 · n_h · c)",
          evaluated: `${formatTflops(result.csaLayer.q)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.q * fullLayers)}`
        },
        {
          label: "Full layer KV proj FLOPs",
          expression: "2 · S · D · n_kv · c · 2  (K+V)",
          evaluated: `${formatTflops(result.csaLayer.kvProj)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.kvProj * fullLayers)}`
        },
        {
          label: "Full core attention FLOPs (causal ≈ S²/2)",
          expression: "2 · S² · n_h · c  (causal_factor=2)",
          evaluated: `${formatTflops(result.csaLayer.core)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.core * fullLayers)}`
        },
        {
          label: "Full output proj FLOPs",
          expression: "2 · S · n_h · c · D",
          evaluated: `${formatTflops(result.csaLayer.output)} per layer × ${fullLayers} = ${formatTflops(result.csaLayer.output * fullLayers)}`
        },
        {
          label: "Linear in_proj_qkv + in_proj_z FLOPs",
          expression: "2·S·D·(2·key_dim+value_dim) + 2·S·D·value_dim",
          evaluated: `${formatTflops(result.hcaLayer.q)} per layer × ${linearLayers} = ${formatTflops(result.hcaLayer.q * linearLayers)}`
        },
        {
          label: "Linear in_proj_a+b FLOPs",
          expression: "2 · S · D · 2 · n_v_heads",
          evaluated: `${formatTflops(result.hcaLayer.kvProj)} per layer × ${linearLayers} = ${formatTflops(result.hcaLayer.kvProj * linearLayers)}`
        },
        {
          label: "Linear Conv1D FLOPs",
          expression: "2 · kernel · S · conv_dim",
          evaluated: `${formatTflops(result.hcaLayer.compressor)} per layer × ${linearLayers} = ${formatTflops(result.hcaLayer.compressor * linearLayers)}`
        },
        {
          label: "Linear gated delta scan FLOPs",
          expression: `2 · S · n_v_heads(${n_vhL}) · c_kL(${c_kL}) · c_vL(${c_vL})`,
          evaluated: `${formatTflops(result.hcaLayer.core)} per layer × ${linearLayers} = ${formatTflops(result.hcaLayer.core * linearLayers)}`
        },
        {
          label: "Linear output proj FLOPs",
          expression: "2 · S · value_dim · D",
          evaluated: `${formatTflops(result.hcaLayer.output)} per layer × ${linearLayers} = ${formatTflops(result.hcaLayer.output * linearLayers)}`
        },
        {
          label: ffnLabel,
          expression: ffnExpression,
          evaluated: `${formatTflops(result.csaLayer.moe)} per layer × ${model.decoderLayers} = ${formatTflops(result.csaLayer.moe * model.decoderLayers)}`
        },
        {
          label: "Full layer total",
          expression: isDenseHybrid
            ? "F_Q+gate + F_KV + F_core + F_O + F_FFN"
            : "F_Q+gate + F_KV + F_core + F_O + F_MoE",
          evaluated: `${formatTflops(result.csaLayer.total)} per layer × ${fullLayers} = ${formatTflops(fullTotal)}`
        },
        {
          label: "Linear layer total",
          expression: isDenseHybrid
            ? "F_inproj + F_conv + F_scan + F_O + F_FFN"
            : "F_inproj + F_conv + F_scan + F_O + F_MoE",
          evaluated: `${formatTflops(result.hcaLayer.total)} per layer × ${linearLayers} = ${formatTflops(linearTotal)}`
        },
        {
          label: "Prefill Total FLOPs",
          expression: "L_full · F_full + L_linear · F_linear",
          evaluated: `${formatTflops(result.prefillFlops)}`
        }
      ]
    },
    {
      category: "decode",
      rows: [
        {
          label: "Decode full attention visible length",
          expression: "L_kv_decode(full) = S_ctx",
          evaluated: `${result.decodeCsaLkv} tokens`
        },
        {
          label: "Decode linear attention visible length",
          expression: "L_kv_decode(linear) = 0 (recurrent state, no KV cache)",
          evaluated: "0 tokens (recurrent scan)"
        },
        {
          label: "Full attn bytes per token",
          expression: "B · 2 · n_kv · S_ctx · c · bytes_per_elem (K+V read)",
          evaluated: `${formatMb(result.decodeCsaBytesPerToken)} x ${fullLayers} = ${formatMb(result.decodeCsaBytesPerToken * fullLayers)}`
        },
        {
          label: "Linear attn bytes per token",
          expression: "B · (conv_state + recurrent_state) · bytes_per_elem",
          evaluated: `${formatMb(result.decodeHcaBytesPerToken)} x ${linearLayers} = ${formatMb(result.decodeHcaBytesPerToken * linearLayers)}`
        },
        {
          label: "Decode cache bytes per token",
          expression: "L_full · B_full + L_linear · B_linear",
          evaluated: formatMb(result.decodeCacheBytes)
        },
        {
          label: "Decode weight bytes per token",
          expression: "B_weights = N_non · bpw + N_exp · (k/E) · bpe",
          evaluated: formatMb(result.decodeWeightBytes)
        },
        {
          label: "Total decode bytes per token",
          expression: "B_decode = B_weights + B_cache",
          evaluated: `${formatMb(result.decodeWeightBytes)} + ${formatMb(result.decodeCacheBytes)} = ${formatMb(result.decodeBytes)}`
        },
        {
          label: "Decode compute FLOPs per token (full+linear+MoE)",
          expression: "per-section summed",
          evaluated: formatGflops(result.decodeComputeFlopsPerToken)
        },
        {
          label: "Decode compute ceiling",
          expression: "effective_compute / FLOPs_per_token",
          evaluated: `${result.decodeComputeTps.toFixed(2)} tokens/s`
        },
        {
          label: "Decode bandwidth ceiling",
          expression: "effective_bandwidth / bytes_per_token",
          evaluated: `${result.decodeBandwidthTps.toFixed(2)} tokens/s`
        },
        {
          label: "Effective Decode TPS",
          expression: "min(compute_ceiling, bandwidth_ceiling)",
          evaluated: `${result.decodeTps.toFixed(2)} tokens/s`
        }
      ]
    },
    {
      category: "memory",
      rows: [
        {
          label: "Weight memory",
          expression: "M_weights = N_non · bpw + N_exp · bpe",
          evaluated: `${result.weightGb.toFixed(2)} GB`,
          sourceLabel: "权重文件出处",
          sourceUrl: model.weightSourceUrl
        },
        {
          label: "Full attention KV cache (per layer)",
          expression: "B · 2 · n_kv · S_ctx · c · bytes_per_elem",
          evaluated: `${formatMb(result.persistentCsaCacheBytes)} × ${fullLayers} = ${formatMb(result.persistentCsaCacheTotalBytes)}`
        },
        {
          label: "Linear attention state (per layer)",
          expression: "B · (conv_state + recurrent_state) · bytes_per_elem",
          evaluated: `${formatMb(result.persistentHcaCacheBytes)} × ${linearLayers} = ${formatMb(result.persistentHcaCacheTotalBytes)}`
        },
        {
          label: "Persistent Decode Cache / State",
          expression: "M_cache = M_fullKV + M_linearState",
          evaluated: `${result.cacheGb.toFixed(3)} GB`
        },
        {
          label: "Single-step Temp Peak",
          expression: "B · 2 · 2 · n_h · S_ctx · c (repeat_kv)",
          evaluated: `${formatMb(result.tmpPeakBytes)}; L_kv = ${result.tmpPeakLkv}`
        },
        {
          label: "Runtime overhead",
        expression: "user runtime-overhead assumption",
          evaluated: `${result.overheadGb.toFixed(2)} GB`
        },
        {
          label: "Decode base memory",
          expression: "M_weights + M_cache",
          evaluated: `${(result.weightGb + result.cacheGb).toFixed(2)} GB`
        },
        {
          label: "Decode peak memory",
          expression: "M_weights + M_cache + M_tmp + M_overhead",
          evaluated: `${result.totalRuntimeMemoryGb.toFixed(2)} GB`
        }
      ]
    }
  ];
}

export function calculatePerformanceResult(
  model: ModelDefinition,
  platform: PlatformInput,
  workload: WorkloadInput
): PerformanceResult {
  const computeFn =
    model.formulaStrategyId === "dense-decoder-transformer" ||
    model.formulaStrategyId === "dense-decoder-moe"
      ? computeDenseFullResult
      : model.formulaStrategyId === "deepseek-v4-compressed-moe"
        ? computeFullResult
        : model.formulaStrategyId === "hybrid-linear-moe" ||
            model.formulaStrategyId === "hybrid-linear-dense"
          ? computeHybridLinearMoeResult
          : null;

  if (!computeFn) {
    throw new Error(`Unsupported formula strategy: ${model.formulaStrategyId}`);
  }

  const tokenSweepSeries: TokenSweepPoint[] = [];

  for (
    let tokenLength = workload.tokenRangeStart;
    tokenLength <= workload.tokenRangeEnd;
    tokenLength += workload.tokenRangeStep
  ) {
    const sweepWorkload = { ...workload, prefillTokenLength: tokenLength };
    const pointResult = computeFn(model, platform, sweepWorkload, tokenLength);
    tokenSweepSeries.push({
      tokenLength,
      prefillTps: pointResult.prefillTps,
      decodeTps: pointResult.decodeTps,
      ttftMs: pointResult.ttftMs,
      totalRuntimeMemoryGb: pointResult.totalRuntimeMemoryGb,
      prefillBottleneck: pointResult.prefillBottleneck,
      decodeBottleneck: pointResult.decodeBottleneck
    });
  }

  const activeResult = computeFn(
    model,
    platform,
    workload,
    workload.prefillTokenLength
  );

  return {
    summary: {
      ttftMs: activeResult.ttftMs,
      prefillTps: activeResult.prefillTps,
      decodeTps: activeResult.decodeTps,
      totalRuntimeMemoryGb: activeResult.totalRuntimeMemoryGb,
      prefillBottleneck: activeResult.prefillBottleneck,
      decodeBottleneck: activeResult.decodeBottleneck,
      memoryFitsCapacity: activeResult.memoryFitsCapacity
    },
    comparisonRows: buildComparisonRows(activeResult),
    memoryBreakdown: [
      {
        key: "weights",
        label: "Weights",
        valueGb: activeResult.weightGb
      },
      {
        key: "persistentDecodeCache",
        label: "Persistent Decode Cache",
        valueGb: activeResult.cacheGb
      },
      {
        key: "peakTempWorkingSet",
        label: "Peak Temp Working Set",
        valueGb: activeResult.tmpPeakGb
      },
      {
        key: "runtimeOverhead",
        label: "Runtime Overhead",
        valueGb: activeResult.overheadGb
      },
      {
        key: "estimatedTotal",
        label: "Estimated Total",
        valueGb: activeResult.totalRuntimeMemoryGb
      }
    ],
    intermediateMetrics: buildIntermediateMetrics(
      model,
      platform,
      workload,
      activeResult
    ),
    formulaTrace: buildFormulaTrace(model, workload, activeResult),
    tokenSweepSeries
  };
}
