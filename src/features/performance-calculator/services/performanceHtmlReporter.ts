import type { ModelDefinition } from "../../../domain/model/types";
import type { PerformanceResult } from "../../../domain/performance/types";
import type { CalculationSnapshot } from "../state/useCalculatorState";

export type PerformanceHtmlReportInput = {
  model: ModelDefinition;
  snapshot: CalculationSnapshot;
  result: PerformanceResult;
  exportedAt?: Date;
};

type ReportRow = {
  item: string;
  value: string;
  notes: string;
};

type ReportGroup = {
  type: string;
  tone:
    | "config"
    | "ffn"
    | "full"
    | "sliding"
    | "linear"
    | "compressed"
    | "indexer"
    | "result"
    | "trace";
  rows: ReportRow[];
};

const FLOPS_PER_GFLOP = 1_000_000_000;

function escapeHtml(value: unknown) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatNumber(value: number, maximumFractionDigits = 2) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits,
    useGrouping: true
  }).format(value);
}

function formatGflops(value: number) {
  return `${formatNumber(value / FLOPS_PER_GFLOP, 4)} GFLOPs`;
}

function buildQwen36ComputeGroups(
  model: ModelDefinition,
  snapshot: CalculationSnapshot
): ReportGroup[] {
  const S = snapshot.workload.prefillTokenLength;
  const D = model.hiddenSize;
  const fullLayers = model.fullAttentionLayerCount ?? 0;
  const linearLayers = model.linearAttentionLayerCount ?? 0;
  const nHeads = model.attentionHeads;
  const nKvHeads = model.kvHeads;
  const headDim = model.headDim;
  const keyHeads = model.linearNumKeyHeads ?? 0;
  const keyHeadDim = model.linearKeyHeadDim ?? 0;
  const valueHeads = model.linearNumValueHeads ?? 0;
  const valueHeadDim = model.linearValueHeadDim ?? 0;
  const convKernel = model.linearConvKernelDim ?? 0;
  const keyDim = keyHeads * keyHeadDim;
  const valueDim = valueHeads * valueHeadDim;
  const convDim = 2 * keyDim + valueDim;
  const isDense = model.formulaStrategyId === "hybrid-linear-dense";
  const intermediate = isDense
    ? (model.intermediateSize ?? model.moeIntermediateSize)
    : model.moeIntermediateSize;
  const activeExpertCount = isDense ? 1 : model.activeExperts + 1;

  const ffnPerTokenPerLayer = 6 * D * intermediate * activeExpertCount;
  const ffnAllLayers = ffnPerTokenPerLayer * model.decoderLayers;

  const fullQGate = 2 * D * (2 * nHeads * headDim);
  const fullK = 2 * D * nKvHeads * headDim;
  const fullV = fullK;
  const fullOutput = 2 * nHeads * headDim * D;
  const fullCorePerToken = 2 * S * nHeads * headDim;
  const fullAttentionPerLayer = fullQGate + fullK + fullV + fullOutput + fullCorePerToken;
  const fullAttentionAllLayers = fullAttentionPerLayer * fullLayers;

  const linearQkv = 2 * D * convDim;
  const linearZ = 2 * D * valueDim;
  const linearAB = 2 * D * (2 * valueHeads);
  const linearConv = 2 * convKernel * convDim;
  const linearScan = 2 * valueHeads * keyHeadDim * valueHeadDim;
  const linearOutput = 2 * valueDim * D;
  const linearAttentionPerLayer =
    linearQkv + linearZ + linearAB + linearConv + linearScan + linearOutput;
  const linearAttentionAllLayers = linearAttentionPerLayer * linearLayers;
  const prefillPerToken = ffnAllLayers + fullAttentionAllLayers + linearAttentionAllLayers;

  const groups: ReportGroup[] = [
    {
      type: "Model Config",
      tone: "config",
      rows: [
        { item: "hidden_size H", value: formatNumber(D, 0), notes: "模型隐藏维度" },
        { item: "num_layers", value: formatNumber(model.decoderLayers, 0), notes: "Decoder 总层数" },
        { item: "full_attn_layers", value: formatNumber(fullLayers, 0), notes: "Full GQA 层数" },
        { item: "linear_attn_layers", value: formatNumber(linearLayers, 0), notes: "Gated DeltaNet 层数" },
        { item: "attn_heads", value: formatNumber(nHeads, 0), notes: "Query attention heads" },
        { item: "head_dim", value: formatNumber(headDim, 0), notes: "Full attention 单头维度" },
        { item: "kv_heads", value: formatNumber(nKvHeads, 0), notes: "Full GQA 的 K/V heads" },
        { item: "intermediate_size", value: formatNumber(intermediate, 0), notes: isDense ? "Dense SwiGLU 中间维度" : "单个专家中间维度" },
        ...(isDense
          ? []
          : [
              { item: "routed_experts", value: formatNumber(model.moeExperts, 0), notes: "每层 routed experts 总数" },
              { item: "active_experts / token", value: formatNumber(model.activeExperts, 0), notes: "每 token 激活的 routed experts，不含 shared expert" },
              { item: "shared_experts", value: "1", notes: "每 token 同时执行的 shared expert" }
            ]),
        { item: "prefill_tokens N", value: formatNumber(S, 0), notes: "本报告使用的 Prompt Token Length" }
      ]
    },
    {
      type: isDense ? "Dense FFN\n(SwiGLU, every layer)" : "MoE FFN\n(active experts, every layer)",
      tone: "ffn",
      rows: [
        { item: "Gate projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "Up projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "Down projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "FFN total / layer / token", value: formatGflops(ffnPerTokenPerLayer), notes: `6 × H × I × ${activeExpertCount}` },
        { item: "FFN total / token", value: formatGflops(ffnAllLayers), notes: `FFN / layer / token × ${model.decoderLayers} layers` }
      ]
    },
    {
      type: "Gated Attention\n(Full GQA)",
      tone: "full",
      rows: [
        { item: "Project Q + Gate / layer / token", value: formatGflops(fullQGate), notes: "2 × H × (2 × attn_heads × head_dim)" },
        { item: "Project K / layer / token", value: formatGflops(fullK), notes: "2 × H × kv_heads × head_dim" },
        { item: "Project V / layer / token", value: formatGflops(fullV), notes: "2 × H × kv_heads × head_dim" },
        { item: "Project O / layer / token", value: formatGflops(fullOutput), notes: "2 × attn_heads × head_dim × H" },
        { item: "QKᵀ + AV / layer / token", value: formatGflops(fullCorePerToken), notes: "2 × N × attn_heads × head_dim；按 causal prefill 口径" },
        { item: "Full attention total / layer / token", value: formatGflops(fullAttentionPerLayer), notes: "Q/Gate + K + V + O + attention core" },
        { item: "Full attention total / token", value: formatGflops(fullAttentionAllLayers), notes: `Full attention / layer / token × ${fullLayers} layers` }
      ]
    },
    {
      type: "Gated DeltaNet Attention\n(Linear Attention)",
      tone: "linear",
      rows: [
        { item: "in_proj_qkv / layer / token", value: formatGflops(linearQkv), notes: "2 × H × (2 × key_dim + value_dim)" },
        { item: "in_proj_z / layer / token", value: formatGflops(linearZ), notes: "2 × H × value_dim" },
        { item: "in_proj_a + in_proj_b / layer / token", value: formatGflops(linearAB), notes: "2 × H × (2 × value_heads)" },
        { item: "Depthwise Conv1D / layer / token", value: formatGflops(linearConv), notes: "2 × kernel × conv_dim" },
        { item: "Gated delta rule / layer / token", value: formatGflops(linearScan), notes: "2 × value_heads × key_head_dim × value_head_dim" },
        { item: "Output projection / layer / token", value: formatGflops(linearOutput), notes: "2 × value_dim × H" },
        { item: "Linear attention total / layer / token", value: formatGflops(linearAttentionPerLayer), notes: "Input projections + Conv1D + delta scan + output projection" },
        { item: "Linear attention total / token", value: formatGflops(linearAttentionAllLayers), notes: `Linear attention / layer / token × ${linearLayers} layers` }
      ]
    },
    {
      type: "Prefill Total",
      tone: "result",
      rows: [
        { item: "Theoretical compute / token", value: formatGflops(prefillPerToken), notes: "FFN + Full GQA + Gated DeltaNet" },
        { item: "Theoretical compute / request", value: `${formatNumber((prefillPerToken * S) / 1e12, 3)} TFLOPs`, notes: "Theoretical compute / token × N" }
      ]
    }
  ];

  return groups;
}

function buildDenseDecoderComputeGroups(
  model: ModelDefinition,
  snapshot: CalculationSnapshot
): ReportGroup[] {
  const S = snapshot.workload.prefillTokenLength;
  const D = model.hiddenSize;
  const nHeads = model.attentionHeads;
  const slidingLayers = model.slidingAttentionLayerCount ?? model.slidingLayerCount;
  const fullLayers = model.fullAttentionLayerCount ?? 0;
  const slidingHeadDim = model.headDim;
  const fullHeadDim = model.globalHeadDim ?? model.headDim;
  const slidingKvHeads = model.kvHeads;
  const fullKvHeads = model.numGlobalKeyValueHeads ?? model.kvHeads;
  const fullHasVProjection = !(model.attentionKEqV ?? false);
  const isMoe = model.formulaStrategyId === "dense-decoder-moe";
  const intermediate = isMoe
    ? model.moeIntermediateSize
    : (model.intermediateSize ?? model.moeIntermediateSize);
  const activeExpertCount = isMoe ? model.activeExperts + 1 : 1;

  const ffnPerLayer = 6 * D * intermediate * activeExpertCount;
  const ffnAllLayers = ffnPerLayer * model.decoderLayers;

  const slidingQ = 2 * D * nHeads * slidingHeadDim;
  const slidingKv = 2 * D * slidingKvHeads * slidingHeadDim * 2;
  const slidingCore = 2 * model.slidingWindow * nHeads * slidingHeadDim;
  const slidingOutput = 2 * nHeads * slidingHeadDim * D;
  const slidingPerLayer = slidingQ + slidingKv + slidingCore + slidingOutput;
  const slidingAllLayers = slidingPerLayer * slidingLayers;

  const fullQ = 2 * D * nHeads * fullHeadDim;
  const fullKv =
    2 * D * fullKvHeads * fullHeadDim * (fullHasVProjection ? 2 : 1);
  const fullCore = 2 * S * nHeads * fullHeadDim;
  const fullOutput = 2 * nHeads * fullHeadDim * D;
  const fullPerLayer = fullQ + fullKv + fullCore + fullOutput;
  const fullAllLayers = fullPerLayer * fullLayers;
  const prefillPerToken = ffnAllLayers + slidingAllLayers + fullAllLayers;

  return [
    {
      type: "Model Config",
      tone: "config",
      rows: [
        { item: "hidden_size H", value: formatNumber(D, 0), notes: "模型隐藏维度" },
        { item: "num_layers", value: formatNumber(model.decoderLayers, 0), notes: "Decoder 总层数" },
        { item: "sliding_attn_layers", value: formatNumber(slidingLayers, 0), notes: "Sliding-window attention 层数" },
        { item: "full_attn_layers", value: formatNumber(fullLayers, 0), notes: "Full attention 层数" },
        { item: "sliding_window", value: formatNumber(model.slidingWindow, 0), notes: "Sliding attention 可见窗口" },
        { item: "attention_heads", value: formatNumber(nHeads, 0), notes: "Query heads" },
        { item: "sliding / full head_dim", value: `${slidingHeadDim} / ${fullHeadDim}`, notes: "局部与全局注意力 head dimension" },
        { item: "sliding / full kv_heads", value: `${slidingKvHeads} / ${fullKvHeads}`, notes: "局部与全局注意力 KV heads" },
        { item: "intermediate_size", value: formatNumber(intermediate, 0), notes: isMoe ? "单个专家中间维度" : "Dense FFN 中间维度" },
        ...(isMoe
          ? [
              { item: "routed_experts", value: formatNumber(model.moeExperts, 0), notes: "每层 routed experts 总数" },
              { item: "active_experts / token", value: formatNumber(model.activeExperts, 0), notes: "每 token 激活 routed experts 数" },
              { item: "shared_experts", value: "1", notes: "当前计算引擎包含的 shared expert" }
            ]
          : []),
        { item: "prefill_tokens N", value: formatNumber(S, 0), notes: "当前 Prompt Token Length" }
      ]
    },
    {
      type: isMoe ? "MoE FFN\n(active + shared experts)" : `Dense FFN\n(${model.hiddenActivation ?? "activation"})`,
      tone: "ffn",
      rows: [
        { item: "Gate projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "Up projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "Down projection / layer / token", value: formatGflops(2 * D * intermediate * activeExpertCount), notes: `2 × H × I × ${activeExpertCount}` },
        { item: "FFN total / layer / token", value: formatGflops(ffnPerLayer), notes: `6 × H × I × ${activeExpertCount}` },
        { item: "FFN total / token", value: formatGflops(ffnAllLayers), notes: `FFN / layer / token × ${model.decoderLayers} layers` }
      ]
    },
    {
      type: "Sliding-window Attention",
      tone: "sliding",
      rows: [
        { item: "Project Q / layer / token", value: formatGflops(slidingQ), notes: "2 × H × attention_heads × head_dim" },
        { item: "Project K + V / layer / token", value: formatGflops(slidingKv), notes: "2 × H × kv_heads × head_dim × 2" },
        { item: "QKᵀ + AV / layer / token", value: formatGflops(slidingCore), notes: "2 × sliding_window × attention_heads × head_dim" },
        { item: "Project O / layer / token", value: formatGflops(slidingOutput), notes: "2 × attention_heads × head_dim × H" },
        { item: "Sliding attention total / layer / token", value: formatGflops(slidingPerLayer), notes: "Q + K/V + attention core + O" },
        { item: "Sliding attention total / token", value: formatGflops(slidingAllLayers), notes: `Sliding attention / layer / token × ${slidingLayers} layers` }
      ]
    },
    {
      type: "Full Attention",
      tone: "full",
      rows: [
        { item: "Project Q / layer / token", value: formatGflops(fullQ), notes: "2 × H × attention_heads × global_head_dim" },
        { item: fullHasVProjection ? "Project K + V / layer / token" : "Shared K=V projection / layer / token", value: formatGflops(fullKv), notes: `2 × H × full_kv_heads × global_head_dim × ${fullHasVProjection ? 2 : 1}` },
        { item: "QKᵀ + AV / layer / token", value: formatGflops(fullCore), notes: "2 × N × attention_heads × global_head_dim；causal prefill" },
        { item: "Project O / layer / token", value: formatGflops(fullOutput), notes: "2 × attention_heads × global_head_dim × H" },
        { item: "Full attention total / layer / token", value: formatGflops(fullPerLayer), notes: "Q + K/V + attention core + O" },
        { item: "Full attention total / token", value: formatGflops(fullAllLayers), notes: `Full attention / layer / token × ${fullLayers} layers` }
      ]
    },
    {
      type: "Prefill Total",
      tone: "result",
      rows: [
        { item: "Theoretical compute / token", value: formatGflops(prefillPerToken), notes: "FFN + Sliding Attention + Full Attention" },
        { item: "Theoretical compute / request", value: `${formatNumber((prefillPerToken * S) / 1e12, 3)} TFLOPs`, notes: "Theoretical compute / token × N" }
      ]
    }
  ];
}

function buildCompressedMoeComputeGroups(
  model: ModelDefinition,
  snapshot: CalculationSnapshot
): ReportGroup[] {
  const S = snapshot.workload.prefillTokenLength;
  const D = model.hiddenSize;
  const nHeads = model.attentionHeads;
  const headDim = model.headDim;
  const ffnPerLayer =
    6 * D * model.moeIntermediateSize * (model.activeExperts + 1);
  const ffnAllLayers = ffnPerLayer * model.decoderLayers;
  const qProjection =
    2 * (D * model.qLoraRank + model.qLoraRank * nHeads * headDim);
  const kvProjection = 2 * D * headDim;
  const outputProjection =
    2 * (nHeads * headDim * model.oLoraRank + model.oGroups * model.oLoraRank * D);
  const indexerLinear =
    8 * D * model.indexHeadDim +
    2 * model.qLoraRank * model.indexHeads * model.indexHeadDim +
    2 * D * model.indexHeads;
  const indexerAttention =
    (S * model.indexHeads * model.indexHeadDim) / model.csaCompressRate;

  const buildAttentionGroup = (
    type: string,
    tone: ReportGroup["tone"],
    layerCount: number,
    visibleLength: number,
    compressor: number,
    includeIndexer: boolean
  ): { group: ReportGroup; allLayers: number } => {
    const core = 4 * visibleLength * nHeads * headDim;
    const perLayer =
      qProjection +
      kvProjection +
      core +
      compressor +
      (includeIndexer ? indexerLinear + indexerAttention : 0) +
      outputProjection;
    const rows: ReportRow[] = [
      { item: "Q LoRA projection / layer / token", value: formatGflops(qProjection), notes: "2 × (H × r_q + r_q × attention_heads × head_dim)" },
      { item: "Compressed KV projection / layer / token", value: formatGflops(kvProjection), notes: "2 × H × head_dim" },
      { item: "Attention core / layer / token", value: formatGflops(core), notes: `4 × L_visible(${formatNumber(visibleLength, 0)}) × attention_heads × head_dim` },
      ...(compressor > 0
        ? [{ item: "Compressor / layer / token", value: formatGflops(compressor), notes: type.startsWith("CSA") ? "8 × H × head_dim" : "4 × H × head_dim" }]
        : []),
      ...(includeIndexer
        ? [
            { item: "Indexer projections / layer / token", value: formatGflops(indexerLinear), notes: "Linear query/key/score projections" },
            { item: "Indexer attention / layer / token", value: formatGflops(indexerAttention), notes: "N × index_heads × index_head_dim / compression_rate" }
          ]
        : []),
      { item: "Grouped O projection / layer / token", value: formatGflops(outputProjection), notes: "2 × (attention_heads × head_dim × r_o + o_groups × r_o × H)" },
      { item: `${type} total / layer / token`, value: formatGflops(perLayer), notes: "该注意力板块算子合计，不含 MoE FFN" },
      { item: `${type} total / token`, value: formatGflops(perLayer * layerCount), notes: `${type} / layer / token × ${layerCount} layers` }
    ];
    return { group: { type, tone, rows }, allLayers: perLayer * layerCount };
  };

  const sliding = buildAttentionGroup(
    "Sliding Attention",
    "sliding",
    model.slidingLayerCount,
    model.slidingWindow,
    0,
    false
  );
  const csa = buildAttentionGroup(
    "CSA Attention\n(Compressed + Selected)",
    "compressed",
    model.csaLayerCount,
    model.slidingWindow + model.indexTopk,
    8 * D * headDim,
    true
  );
  const hca = buildAttentionGroup(
    "HCA Attention\n(Hierarchical Compression)",
    "indexer",
    model.hcaLayerCount,
    model.slidingWindow + Math.ceil(S / (2 * model.hcaCompressRate)),
    4 * D * headDim,
    false
  );
  const prefillPerToken =
    ffnAllLayers + sliding.allLayers + csa.allLayers + hca.allLayers;

  return [
    {
      type: "Model Config",
      tone: "config",
      rows: [
        { item: "hidden_size H", value: formatNumber(D, 0), notes: "模型隐藏维度" },
        { item: "num_layers", value: formatNumber(model.decoderLayers, 0), notes: "Decoder 总层数" },
        { item: "sliding / CSA / HCA layers", value: `${model.slidingLayerCount} / ${model.csaLayerCount} / ${model.hcaLayerCount}`, notes: "三类 attention schedule" },
        { item: "attention_heads / head_dim", value: `${nHeads} / ${headDim}`, notes: "Compressed attention query heads 与维度" },
        { item: "Q / O LoRA rank", value: `${model.qLoraRank} / ${model.oLoraRank}`, notes: "Q 与 grouped O projection 的低秩维度" },
        { item: "routed / active experts", value: `${model.moeExperts} / ${model.activeExperts}`, notes: "总专家数 / 每 token 激活 routed experts" },
        { item: "shared experts", value: "1", notes: "当前计算引擎每 token 额外计入一个 shared expert" },
        { item: "MoE intermediate_size", value: formatNumber(model.moeIntermediateSize, 0), notes: "单专家中间维度" },
        { item: "prefill_tokens N", value: formatNumber(S, 0), notes: "当前 Prompt Token Length" }
      ]
    },
    {
      type: "Sparse MoE FFN\n(active + shared experts)",
      tone: "ffn",
      rows: [
        { item: "Gate projection / layer / token", value: formatGflops(2 * D * model.moeIntermediateSize * (model.activeExperts + 1)), notes: `2 × H × I × ${model.activeExperts + 1}` },
        { item: "Up projection / layer / token", value: formatGflops(2 * D * model.moeIntermediateSize * (model.activeExperts + 1)), notes: `2 × H × I × ${model.activeExperts + 1}` },
        { item: "Down projection / layer / token", value: formatGflops(2 * D * model.moeIntermediateSize * (model.activeExperts + 1)), notes: `2 × H × I × ${model.activeExperts + 1}` },
        { item: "MoE total / layer / token", value: formatGflops(ffnPerLayer), notes: `6 × H × I × ${model.activeExperts + 1}` },
        { item: "MoE total / token", value: formatGflops(ffnAllLayers), notes: `MoE / layer / token × ${model.decoderLayers} layers` }
      ]
    },
    sliding.group,
    csa.group,
    hca.group,
    {
      type: "Prefill Total",
      tone: "result",
      rows: [
        { item: "Theoretical compute / token", value: formatGflops(prefillPerToken), notes: "MoE + Sliding + CSA + HCA" },
        { item: "Theoretical compute / request", value: `${formatNumber((prefillPerToken * S) / 1e12, 3)} TFLOPs`, notes: "Theoretical compute / token × N" }
      ]
    }
  ];
}

function buildReportGroups(input: PerformanceHtmlReportInput): ReportGroup[] {
  switch (input.model.formulaStrategyId) {
    case "hybrid-linear-dense":
    case "hybrid-linear-moe":
      return buildQwen36ComputeGroups(input.model, input.snapshot);
    case "dense-decoder-transformer":
    case "dense-decoder-moe":
      return buildDenseDecoderComputeGroups(input.model, input.snapshot);
    case "deepseek-v4-compressed-moe":
      return buildCompressedMoeComputeGroups(input.model, input.snapshot);
    default: {
      const exhaustiveCheck: never = input.model.formulaStrategyId;
      throw new Error(`Unsupported HTML report formula strategy: ${exhaustiveCheck}`);
    }
  }
}

function renderRows(groups: ReportGroup[]) {
  return groups
    .map((group) =>
      group.rows
        .map(
          (row, rowIndex) => `<tr class="tone-${group.tone}">
            ${rowIndex === 0 ? `<th class="group-cell" rowspan="${group.rows.length}">${escapeHtml(group.type).replace(/\n/g, "<br>")}</th>` : ""}
            <th class="item-cell">${escapeHtml(row.item)}</th>
            <td class="value-cell">${escapeHtml(row.value)}</td>
            <td class="notes-cell">${escapeHtml(row.notes)}</td>
          </tr>`
        )
        .join("\n")
    )
    .join("\n");
}

export function buildPerformanceHtmlReport(input: PerformanceHtmlReportInput) {
  const exportedAt = input.exportedAt ?? new Date();
  const groups = buildReportGroups(input);
  const { model, snapshot, result } = input;
  const title = `Derivation of Theoretical Compute for ${model.displayName} Prefill`;

  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(title)}</title>
  <style>
    :root { color-scheme: light; font-family: Inter, "Segoe UI", "Microsoft YaHei", sans-serif; color: #17243a; background: #eef2f7; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 32px; }
    main { max-width: 1480px; margin: 0 auto; padding: 34px; background: #fff; box-shadow: 0 18px 60px rgba(23,36,58,.12); }
    h1 { margin: 0; padding-bottom: 14px; color: #1557a6; font-size: clamp(24px, 3vw, 38px); border-bottom: 3px solid #3971ad; }
    .meta { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; margin: 22px 0; }
    .meta div { padding: 12px 14px; background: #f7f9fc; border: 1px solid #d7dfea; border-radius: 8px; }
    .meta span { display: block; color: #64748b; font-size: 12px; }
    .meta strong { display: block; margin-top: 4px; font-size: 15px; overflow-wrap: anywhere; }
    .report-note { margin: 0 0 18px; padding: 12px 14px; color: #475569; background: #eff6ff; border-left: 4px solid #2563eb; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 13px; }
    col.type { width: 19%; } col.item { width: 25%; } col.value { width: 18%; } col.notes { width: 38%; }
    th, td { padding: 8px 10px; border: 1px solid #aeb8c6; vertical-align: middle; }
    thead th { color: #fff; background: #17243a; text-align: center; }
    .group-cell { text-align: center; white-space: normal; font-size: 14px; }
    .item-cell { text-align: left; }
    .value-cell { text-align: right; font-variant-numeric: tabular-nums; font-weight: 700; }
    .notes-cell { color: #334155; }
    .tone-config .group-cell, .tone-config .item-cell { background: #e8edf3; }
    .tone-config .value-cell { background: #f4f6f9; }
    .tone-ffn .group-cell, .tone-ffn .item-cell { background: #bfdbfe; }
    .tone-ffn .value-cell { background: #dbeafe; }
    .tone-full .group-cell, .tone-full .item-cell { background: #f59e0b; color: #2d2108; }
    .tone-full .value-cell { background: #fef3c7; }
    .tone-sliding .group-cell, .tone-sliding .item-cell { background: #38bdf8; color: #082f49; }
    .tone-sliding .value-cell { background: #e0f2fe; }
    .tone-linear .group-cell, .tone-linear .item-cell { background: #84cc16; color: #18310a; }
    .tone-linear .value-cell { background: #dcfce7; }
    .tone-compressed .group-cell, .tone-compressed .item-cell { background: #fb923c; color: #431407; }
    .tone-compressed .value-cell { background: #ffedd5; }
    .tone-indexer .group-cell, .tone-indexer .item-cell { background: #a78bfa; color: #2e1065; }
    .tone-indexer .value-cell { background: #ede9fe; }
    .tone-result .group-cell, .tone-result .item-cell { color: #fff; background: #1557a6; }
    .tone-result .value-cell { color: #0f3d73; background: #dbeafe; }
    .tone-trace .group-cell, .tone-trace .item-cell { background: #e9d5ff; }
    .tone-trace .value-cell { background: #f3e8ff; }
    footer { margin-top: 18px; color: #64748b; font-size: 12px; }
    @media (max-width: 900px) { body { padding: 0; } main { padding: 18px; } .meta { grid-template-columns: 1fr 1fr; } table { font-size: 11px; } th,td { padding: 6px; } }
    @media print { @page { size: A4 landscape; margin: 10mm; } body { padding: 0; background: #fff; } main { max-width: none; padding: 0; box-shadow: none; } tr { break-inside: avoid; } }
  </style>
</head>
<body>
<main>
  <h1>${escapeHtml(title)}</h1>
  <section class="meta" aria-label="计算快照">
    <div><span>Model</span><strong>${escapeHtml(model.displayName)}</strong></div>
    <div><span>Prompt Length</span><strong>${formatNumber(snapshot.workload.prefillTokenLength, 0)} tokens</strong></div>
    <div><span>Prefill TPS / TTFT</span><strong>${formatNumber(result.summary.prefillTps)} tokens/s · ${formatNumber(result.summary.ttftMs)} ms</strong></div>
    <div><span>Exported At</span><strong>${escapeHtml(exportedAt.toLocaleString("zh-CN"))}</strong></div>
  </section>
  <p class="report-note">颜色分块表示该模型的主要运算模块；表内数值基于最近一次成功计算的模型与 Prompt 快照。Projection 行为单层、单 token 理论 FLOPs，块汇总按实际层数累加；这不是实测性能数据。</p>
  <table>
    <colgroup><col class="type"><col class="item"><col class="value"><col class="notes"></colgroup>
    <thead><tr><th>Type</th><th>Calculation Layer / Item</th><th>Value</th><th>Notes (Description / Formula)</th></tr></thead>
    <tbody>${renderRows(groups)}</tbody>
  </table>
  <footer>Generated locally by LLM Perf Calculator. Formula strategy: ${escapeHtml(model.formulaStrategyId)}.</footer>
</main>
</body>
</html>`;
}

export function writePerformanceHtmlReport(
  input: PerformanceHtmlReportInput,
  reportWindow: Window
) {
  const html = buildPerformanceHtmlReport(input);
  reportWindow.document.open();
  reportWindow.document.write(html);
  reportWindow.document.close();
}
