import type { ReactNode } from "react";
import type { FormulaStrategyId, ModelId } from "../../domain/model/types";
import type { FormulaTraceSection } from "../../domain/performance/types";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

type PrefillFormulaGuide = {
  expression: string;
  notes: string[];
  variables: FormulaVariable[];
};

type FormulaVariable = {
  symbol: string;
  meaning: string;
};

const prefillFormulaGuides = {
  "deepseek-v4-compressed-moe": {
    expression: `F_prefill = N_sliding * F_sliding
          + N_csa * F_csa
          + N_hca * F_hca

F_indexer_attn = S^2 * n_h_I * c_I / m_csa
F_moe = 6 * S * D * I * (k + 1)`,
    notes: [
      "总 FLOPs 由 Sliding、CSA 与 HCA 三种层的单层 FLOPs 乘以各自层数后相加。",
      "CSA indexer attention 是随输入长度呈 O(S^2) 增长的独立项。",
      "MoE 按每个 token 激活的 routed experts 与 shared expert 计算。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "N_sliding / N_csa / N_hca", meaning: "Sliding、CSA、HCA 三类 decoder 层的数量。" },
      { symbol: "F_sliding / F_csa / F_hca", meaning: "对应类型单个 decoder 层的 Prefill FLOPs。" },
      { symbol: "F_indexer_attn", meaning: "CSA 索引器注意力的 FLOPs。" },
      { symbol: "S", meaning: "Prompt token length，即参与 Prefill 的 token 数。" },
      { symbol: "n_h_I", meaning: "CSA 索引器的 attention head 数。" },
      { symbol: "c_I", meaning: "CSA 索引器每个 head 的维度。" },
      { symbol: "m_csa", meaning: "CSA 的 token 压缩倍率。" },
      { symbol: "F_moe", meaning: "单层 MoE 前馈网络的 FLOPs。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "I", meaning: "单个专家的中间层维度。" },
      { symbol: "k", meaning: "每个 token 激活的 routed expert 数；k + 1 包含 shared expert。" }
    ]
  },
  "dense-decoder-transformer": {
    expression: `F_prefill = L_sliding * F_sliding
          + L_full * F_full

F_layer = F_Q + F_KV + F_attention
        + F_O + F_MLP
F_MLP = 6 * S * D * I`,
    notes: [
      "总 FLOPs 由 Sliding Attention 层与 Full Attention 层的计算量汇总。",
      "两种层分别展开 Q、KV、Attention、输出投影和 Dense MLP。",
      "Full Attention 核心项随输入长度呈 O(S^2) 增长。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_sliding / L_full", meaning: "Sliding Attention 层数与 Full Attention 层数。" },
      { symbol: "F_sliding / F_full", meaning: "单个 Sliding 层与单个 Full 层的 FLOPs。" },
      { symbol: "F_layer", meaning: "当前类型单个 decoder 层的总 FLOPs。" },
      { symbol: "F_Q / F_KV", meaning: "Query 投影与 Key/Value 投影的 FLOPs。" },
      { symbol: "F_attention", meaning: "Attention 核心计算的 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影的 FLOPs。" },
      { symbol: "F_MLP", meaning: "Dense GeGLU/SwiGLU 前馈网络的 FLOPs。" },
      { symbol: "S", meaning: "Prompt token length。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "I", meaning: "Dense MLP 的 intermediate size。" }
    ]
  },
  "dense-decoder-moe": {
    expression: `F_prefill = L_sliding * F_sliding
          + L_full * F_full

F_layer = F_Q + F_KV + F_attention
        + F_O + F_moe
F_moe = 6 * S * D * I_moe * (k + 1)`,
    notes: [
      "总 FLOPs 由 Sliding Attention 层与 Full Attention 层的计算量汇总。",
      "Attention 路径按层类型分别计算，MoE 路径应用于全部 decoder layers。",
      "MoE 只计算每个 token 实际激活的 routed experts 与 shared expert。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_sliding / L_full", meaning: "Sliding Attention 层数与 Full Attention 层数。" },
      { symbol: "F_sliding / F_full", meaning: "单个 Sliding 层与单个 Full 层的 FLOPs。" },
      { symbol: "F_layer", meaning: "当前类型单个 decoder 层的总 FLOPs。" },
      { symbol: "F_Q / F_KV", meaning: "Query 投影与 Key/Value 投影的 FLOPs。" },
      { symbol: "F_attention", meaning: "Attention 核心计算的 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影的 FLOPs。" },
      { symbol: "F_moe", meaning: "单层 MoE 前馈网络的 FLOPs。" },
      { symbol: "S", meaning: "Prompt token length。" },
      { symbol: "D", meaning: "模型 hidden size。" },
      { symbol: "I_moe", meaning: "单个专家的 intermediate size。" },
      { symbol: "k", meaning: "每个 token 激活的 routed expert 数；k + 1 包含 shared expert。" }
    ]
  },
  "hybrid-linear-moe": {
    expression: `F_prefill = L_full * F_full
          + L_linear * F_linear

F_full = F_Q+gate + F_KV + F_attention
       + F_O + F_moe
F_linear = F_inproj + F_conv + F_scan
         + F_O + F_moe`,
    notes: [
      "总 FLOPs 由 Full Attention 层和 Linear Attention 层分别汇总。",
      "Full 层包含二次复杂度 Attention；Linear 层展开输入投影、Conv1D 与 gated delta scan。",
      "MoE 按每个 token 实际激活的 routed experts 与 shared expert 计算。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_full / L_linear", meaning: "Full Attention 层数与 Linear Attention 层数。" },
      { symbol: "F_full / F_linear", meaning: "单个 Full 层与单个 Linear 层的 FLOPs。" },
      { symbol: "F_Q+gate", meaning: "Full Attention 的 Query 与 gate 投影 FLOPs。" },
      { symbol: "F_KV", meaning: "Key/Value 投影 FLOPs。" },
      { symbol: "F_attention", meaning: "Full Attention 核心计算 FLOPs。" },
      { symbol: "F_inproj", meaning: "Linear Attention 输入投影 FLOPs。" },
      { symbol: "F_conv", meaning: "Linear Attention Conv1D FLOPs。" },
      { symbol: "F_scan", meaning: "Gated delta scan 递推计算 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影 FLOPs。" },
      { symbol: "F_moe", meaning: "MoE 前馈网络 FLOPs。" }
    ]
  },
  "hybrid-linear-dense": {
    expression: `F_prefill = L_full * F_full
          + L_linear * F_linear

F_full = F_Q+gate + F_KV + F_attention
       + F_O + F_FFN
F_linear = F_inproj + F_conv + F_scan
         + F_O + F_FFN`,
    notes: [
      "总 FLOPs 由 Full Attention 层和 Linear Attention 层分别汇总。",
      "Full 层包含二次复杂度 Attention；Linear 层展开输入投影、Conv1D 与 gated delta scan。",
      "该策略使用 Dense SwiGLU FFN，不使用 routed-expert MoE 项。"
    ],
    variables: [
      { symbol: "F_prefill", meaning: "整个输入序列完成 Prefill 所需的总浮点运算量。" },
      { symbol: "L_full / L_linear", meaning: "Full Attention 层数与 Linear Attention 层数。" },
      { symbol: "F_full / F_linear", meaning: "单个 Full 层与单个 Linear 层的 FLOPs。" },
      { symbol: "F_Q+gate", meaning: "Full Attention 的 Query 与 gate 投影 FLOPs。" },
      { symbol: "F_KV", meaning: "Key/Value 投影 FLOPs。" },
      { symbol: "F_attention", meaning: "Full Attention 核心计算 FLOPs。" },
      { symbol: "F_inproj", meaning: "Linear Attention 输入投影 FLOPs。" },
      { symbol: "F_conv", meaning: "Linear Attention Conv1D FLOPs。" },
      { symbol: "F_scan", meaning: "Gated delta scan 递推计算 FLOPs。" },
      { symbol: "F_O", meaning: "Attention 输出投影 FLOPs。" },
      { symbol: "F_FFN", meaning: "Dense SwiGLU 前馈网络 FLOPs。" }
    ]
  }
} satisfies Record<FormulaStrategyId, PrefillFormulaGuide>;

const prefillTpsVariables: FormulaVariable[] = [
  { symbol: "TPS_prefill", meaning: "Prefill 阶段每秒处理的 token 数。" },
  { symbol: "S", meaning: "Prompt token length。" },
  { symbol: "effective_compute", meaning: "平台峰值算力乘以 compute efficiency 后的有效算力。" },
  { symbol: "F_prefill", meaning: "完成整个 Prompt Prefill 的总 FLOPs。" },
  { symbol: "effective_bandwidth", meaning: "内存带宽乘以 bandwidth efficiency 后的有效带宽。" },
  { symbol: "B_prefill", meaning: "完成整个 Prompt Prefill 预计产生的总内存流量。" }
];

const decodeTpsVariables: FormulaVariable[] = [
  { symbol: "TPS_decode", meaning: "Decode 阶段单序列每秒生成的 token 数。" },
  { symbol: "decode_compute_ceiling", meaning: "由单 token FLOPs 和有效算力决定的 Decode 吞吐上限。" },
  { symbol: "decode_bandwidth_ceiling", meaning: "由权重及缓存流量和有效带宽决定的 Decode 吞吐上限。" }
];

const decodeMemoryVariables: FormulaVariable[] = [
  { symbol: "M_decode_total", meaning: "Decode 运行期间预计需要的总显存。" },
  { symbol: "M_weights", meaning: "常驻模型权重占用。" },
  { symbol: "M_decode_cache", meaning: "跨生成 token 持久保存的 KV cache 或线性注意力状态。" },
  { symbol: "M_decode_tmp_peak", meaning: "单个 Decode step 的临时工作集峰值。" },
  { symbol: "M_runtime_overhead", meaning: "运行时框架、CUDA context、allocator 与 kernel workspace 的估算开销。" }
];

const variableSourceBySymbol: Record<string, string> = {
  S: "工作负载输入：Prompt Token Length",
  "N_sliding / N_csa / N_hca": "模型 config.json：各 Attention 层数",
  "L_sliding / L_full": "模型 config.json：Sliding / Full 层数",
  "L_full / L_linear": "模型 config.json：Full / Linear 层数",
  n_h_I: "模型 config.json：Indexer Attention Heads",
  c_I: "模型 config.json：Indexer Head Dimension",
  m_csa: "模型 config.json：CSA Compression Rate",
  D: "模型 config.json：Hidden Size",
  I: "模型 config.json：Intermediate Size",
  I_moe: "模型 config.json：MoE Intermediate Size",
  k: "模型 config.json：Active Experts / Token",
  effective_compute: "平台参数：Compute Throughput × Compute Efficiency",
  effective_bandwidth: "平台参数：Memory Bandwidth × Bandwidth Efficiency",
  B_prefill: "计算器工程估算：Prefill Memory Traffic",
  M_weights: "模型参数量与 Bytes / Weight、Bytes / Expert",
  M_decode_cache: "模型 config.json、输入长度与缓存精度",
  M_decode_tmp_peak: "模型 config.json、输入长度与 Activation Precision",
  M_runtime_overhead: "平台参数：Runtime Overhead"
};

function getVariableSource(symbol: string): string {
  return variableSourceBySymbol[symbol] ?? "由当前公式及其上游变量派生";
}

function FormulaAccordionItem({
  title,
  children
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <details className="formula-accordion">
      <summary>
        <span>{title}</span>
        <span className="formula-accordion__icon" aria-hidden="true" />
      </summary>
      <div className="formula-accordion__content">{children}</div>
    </details>
  );
}

function FormulaBlock({
  title,
  stage,
  expression,
  notes,
  variables,
  trace
}: {
  title: string;
  stage: string;
  expression: string;
  notes: string[];
  variables: FormulaVariable[];
  trace?: FormulaTraceSection;
}) {
  return (
    <article className="formula-block">
      <div className="formula-block__header">
        <div>
          <p className="eyebrow">{stage}</p>
          <h3>{title}</h3>
        </div>
        <span className="source-badge">calculator source</span>
      </div>
      <pre>{expression}</pre>
      <div className="formula-variable-list">
        <h4>变量含义</h4>
        <dl>
          <div className="formula-variable-list__header" aria-hidden="true">
            <span>变量</span>
            <span>含义</span>
            <span>数据来源</span>
          </div>
          {variables.map((variable) => (
            <div key={variable.symbol} className="formula-variable-list__row">
              <dt><code>{variable.symbol}</code></dt>
              <dd>{variable.meaning}</dd>
              <dd className="formula-variable-list__source">
                {getVariableSource(variable.symbol)}
              </dd>
            </div>
          ))}
        </dl>
      </div>
      <ul className="plain-list">
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      {trace ? <FormulaTracePreview trace={trace} /> : null}
    </article>
  );
}

function FormulaTracePreview({ trace }: { trace: FormulaTraceSection }) {
  return (
    <div className="trace-preview">
      <p className="eyebrow">formula trace</p>
      <div className="trace-preview__grid">
        {trace.rows.map((row) => (
          <div key={row.label} className="trace-preview__row">
            <span className="trace-preview__label">{row.label}</span>
            <code>{row.expression}</code>
            {row.sourceUrl ? (
              <a
                className="trace-preview__source"
                href={row.sourceUrl}
                target="_blank"
                rel="noreferrer"
              >
                {row.sourceLabel ?? "来源"} ↗
              </a>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

export function FormulaNotesPage() {
  const {
    state,
    selectedFamily,
    availableFamilies,
    availableModels,
    selectedModel: model,
    result,
    updateModelFamily,
    updateModelId
  } = useCalculatorContext();

  if (!result) {
    return null;
  }

  const prefillTrace = result.formulaTrace.find((trace) => trace.category === "prefill");
  const decodeTrace = result.formulaTrace.find((trace) => trace.category === "decode");
  const memoryTrace = result.formulaTrace.find((trace) => trace.category === "memory");
  const prefillFormulaGuide = prefillFormulaGuides[model.formulaStrategyId];

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Traceable Formulae</p>
          <h2>公式说明</h2>
        </div>
        <p className="page-description">
          该页面将整理 prefill、decode 与内存估算公式及其变量定义。
        </p>
      </div>

      <div className="formula-layout">
        <main className="formula-nav panel">
          <div className="formula-model-selectors">
            <label className="field">
              <span>模型家族</span>
              <select
                value={selectedFamily}
                onChange={(event) => updateModelFamily(event.target.value)}
              >
                {availableFamilies.map((family) => (
                  <option key={family.id} value={family.id}>
                    {family.displayName}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>当前模型</span>
              <select
                value={state.modelId}
                onChange={(event) => updateModelId(event.target.value as ModelId)}
              >
                {availableModels.map((entry) => (
                  <option key={entry.id} value={entry.id}>
                    {entry.displayName}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="formula-accordion-list">
            <FormulaAccordionItem title="Prefill FLOPs">
              <FormulaBlock
                title="Prefill FLOPs"
                stage="prefill"
                expression={prefillFormulaGuide.expression}
                notes={prefillFormulaGuide.notes}
                variables={prefillFormulaGuide.variables}
                trace={prefillTrace}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem title="Prefill TPS">
              <FormulaBlock
                title="Prefill TPS"
                stage="prefill"
                expression={`TPS_prefill = min(
  S * effective_compute / F_prefill,
  S * effective_bandwidth / B_prefill
)`}
                notes={[
                  "effective_compute comes from platform TFLOPS and compute efficiency.",
                  "effective_bandwidth comes from memory bandwidth and bandwidth efficiency.",
                  "Current B_prefill is an engineering estimate and should be tightened later."
                ]}
                variables={prefillTpsVariables}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem title="Decode TPS">
              <FormulaBlock
                title="Decode TPS"
                stage="decode"
                expression={`TPS_decode = min(
  decode_compute_ceiling,
  decode_bandwidth_ceiling
)`}
                notes={[
                  "Decode bandwidth is dominated by visible cache traffic and repeated per-token access.",
                  "The current compute ceiling is an approximation, not a per-kernel trace.",
                  "Weight reads are included in decode bandwidth traffic."
                ]}
                variables={decodeTpsVariables}
                trace={decodeTrace}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem title="Decode Memory">
              <FormulaBlock
                title="Decode Memory"
                stage="memory"
                expression={`M_decode_total ~= M_weights
  + M_decode_cache
  + M_decode_tmp_peak
  + M_runtime_overhead`}
                notes={[
                  "M_weights is resident during decode and must be included.",
                  "M_decode_cache is persistent across generated tokens.",
                  "M_decode_tmp_peak captures the single-step attention working set."
                ]}
                variables={decodeMemoryVariables}
                trace={memoryTrace}
              />
            </FormulaAccordionItem>

          </div>
        </main>

      </div>
    </section>
  );
}
