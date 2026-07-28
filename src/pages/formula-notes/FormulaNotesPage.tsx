import type { ReactNode } from "react";
import type { ModelId } from "../../domain/model/types";
import type { FormulaTraceSection } from "../../domain/performance/types";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

const symbolRows = [
  ["S", "Token length", "Workload", "131072"],
  ["D", "Hidden size", "Model config", "4096 / 7168"],
  ["n_h", "Attention heads", "Model config", "64 / 128"],
  ["c", "Head dimension", "Model config", "512"],
  ["m_csa", "CSA compress rate", "Model config", "4"],
  ["M_weights", "Resident weight memory", "Model definition", "145.82 GB / 799.31 GB"]
];

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
  evaluated,
  trace
}: {
  title: string;
  stage: string;
  expression: string;
  notes: string[];
  evaluated: string;
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
      <ul className="plain-list">
        {notes.map((note) => (
          <li key={note}>{note}</li>
        ))}
      </ul>
      <div className="formula-block__evaluated">
        <span>Current substitution</span>
        <strong>{evaluated}</strong>
      </div>
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
            <strong>{row.evaluated}</strong>
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
                expression={`F_prefill = N_sliding * F_sliding
          + N_csa * F_csa
          + N_hca * F_hca

F_indexer_attn = S^2 * n_h_I * c_I / m_csa
F_moe = 6 * S * D * I * (k + 1)`}
                notes={[
                  "CSA indexer attention is the O(S^2) term.",
                  "MoE and output projection are usually large contributors.",
                  "Norm, RoPE, topk, and Sinkhorn are ignored in the first calculator version."
                ]}
                evaluated={`${result.intermediateMetrics.find((row) => row.key === "prefill_flops")?.value ?? "-"} FLOPs`}
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
                evaluated={`${result.summary.prefillTps.toFixed(2)} tokens/s`}
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
                evaluated={`${result.summary.decodeTps.toFixed(2)} tokens/s`}
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
                evaluated={`${result.summary.totalRuntimeMemoryGb.toFixed(2)} GB`}
                trace={memoryTrace}
              />
            </FormulaAccordionItem>

            <FormulaAccordionItem title="Symbol Table">
              <div className="formula-symbol-table">
                <table className="data-table data-table--compact">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Meaning</th>
                      <th>Source</th>
                      <th>Example</th>
                    </tr>
                  </thead>
                  <tbody>
                    {symbolRows.map(([symbol, meaning, source, example]) => (
                      <tr key={symbol}>
                        <td>{symbol}</td>
                        <td>{meaning}</td>
                        <td>{source}</td>
                        <td>{example}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </FormulaAccordionItem>
          </div>
        </main>

        <aside className="formula-side">
          <article className="panel">
            <h3>当前代入</h3>
            <div className="stack-list">
              <div className="stack-list__row">
                <strong>Model</strong>
                <span>{model.displayName}</span>
              </div>
              <div className="stack-list__row">
                <strong>Prompt Length</strong>
                <span>{state.workload.prefillTokenLength.toLocaleString()}</span>
              </div>
              <div className="stack-list__row">
                <strong>Initial Decode Context</strong>
                <span>{state.workload.prefillTokenLength.toLocaleString()} (from prompt KV cache)</span>
              </div>
              <div className="stack-list__row">
                <strong>Compute</strong>
                <span>
                  {state.platform.computeThroughputTflops} TFLOPS x {state.platform.computeEfficiency}
                </span>
              </div>
              <div className="stack-list__row">
                <strong>Bandwidth</strong>
                <span>
                  {state.platform.memoryBandwidthGbps} GB/s x {state.platform.bandwidthEfficiency}
                </span>
              </div>
              <div className="stack-list__row">
                <strong>Prefill TPS</strong>
                <span>{result.summary.prefillTps.toFixed(2)}</span>
              </div>
              <div className="stack-list__row">
                <strong>Decode TPS</strong>
                <span>{result.summary.decodeTps.toFixed(2)}</span>
              </div>
              <div className="stack-list__row">
                <strong>Memory</strong>
                <span>{result.summary.totalRuntimeMemoryGb.toFixed(2)} GB</span>
              </div>
            </div>
          </article>
        </aside>
      </div>
    </section>
  );
}
