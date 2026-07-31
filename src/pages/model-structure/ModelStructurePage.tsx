import type { ModelDefinition, ModelId } from "../../domain/model/types";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

function formatK(value: number) {
  return value >= 1024 ? `${value / 1024}K` : String(value);
}

function MetricStrip({ model }: { model: ModelDefinition }) {
  const metrics = [
    ["Layers", model.decoderLayers],
    ["Hidden", model.hiddenSize],
    ["Heads", model.attentionHeads],
    ["KV Heads", model.kvHeads],
    ["Context", formatK(model.contextLimit)],
    ["Weights", `${model.estimatedWeightsGb.toFixed(2)} GB`]
  ];

  return (
    <div className="metric-strip">
      {metrics.map(([label, value]) => (
        <div key={label} className="metric-strip__item">
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function StructureFlowDiagram({ model }: { model: ModelDefinition }) {
  const isDense =
    model.architectureKind === "dense-decoder" ||
    model.architectureKind === "dense-decoder-moe";
  const isDenseMoe = model.architectureKind === "dense-decoder-moe";
  const isHybrid =
    model.architectureKind === "hybrid-linear-moe" ||
    model.architectureKind === "hybrid-linear-dense";
  const isHybridDense = model.architectureKind === "hybrid-linear-dense";

  const nodes = isHybrid
    ? [
      {
        label: "input_ids",
        shape: "[B, S]",
        dtype: "int32",
        tone: "neutral" as const
      },
      {
        label: "embed_tokens",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "hidden · √D",
        tone: "blue" as const
      },
      {
        label: `decoder layers (${model.decoderLayers})`,
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: `Gated DeltaNet + Full GQA + ${isHybridDense ? "Dense SwiGLU" : "MoE"}`,
        tone: "green" as const,
        repeat: `x${model.decoderLayers}`
      },
      {
        label: "norm",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "RMSNorm",
        tone: "violet" as const
      },
      {
        label: "lm_head",
        shape: "[B, S, vocab]",
        dtype: `${model.hiddenSize} → vocab (untied)`,
        tone: "blue" as const
      },
      {
        label: "logits",
        shape: "[B, S, vocab]",
        dtype: "float",
        tone: "neutral" as const
      }
    ]
    : isDense
    ? [
      {
        label: "input_ids",
        shape: "[B, S]",
        dtype: "int32",
        tone: "neutral" as const
      },
      {
        label: "embed_tokens",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "hidden · √D",
        tone: "blue" as const
      },
      {
        label: `decoder layers (${model.decoderLayers})`,
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: isDenseMoe ? "GQA/MQA + Routed MoE" : "GQA/MQA + Dense GeGLU",
        tone: "green" as const,
        repeat: `x${model.decoderLayers}`
      },
      {
        label: "norm",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "RMSNorm",
        tone: "violet" as const
      },
      {
        label: "lm_head",
        shape: "[B, S, vocab]",
        dtype: `${model.hiddenSize} → vocab (tied)`,
        tone: "blue" as const
      },
      {
        label: "logits",
        shape: "[B, S, vocab]",
        dtype: "float + softcapping",
        tone: "neutral" as const
      }
    ]
    : [
      {
        label: "input_ids",
        shape: "[B, S]",
        dtype: "int32",
        tone: "neutral" as const
      },
      {
        label: "embed_tokens",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "hidden",
        tone: "blue" as const
      },
      {
        label: `decoder layers (${model.decoderLayers})`,
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "Attention + Compressed Cache + MoE",
        tone: "green" as const,
        repeat: `x${model.decoderLayers}`
      },
      {
        label: "hc_head",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "4 streams -> 1 stream",
        tone: "amber" as const
      },
      {
        label: "norm",
        shape: `[B, S, ${model.hiddenSize}]`,
        dtype: "RMSNorm",
        tone: "violet" as const
      },
      {
        label: "lm_head",
        shape: "[B, S, vocab]",
        dtype: `${model.hiddenSize} -> 129280`,
        tone: "blue" as const
      },
      {
        label: "logits",
        shape: "[B, S, vocab]",
        dtype: "float",
        tone: "neutral" as const
      }
    ];

  return (
    <div className="structure-flow">
      <div className="structure-flow__pipeline">
        {nodes.map((node, index) => (
          <div key={node.label} className="structure-flow__row">
            <div className="structure-flow__repeat">{node.repeat ?? ""}</div>
            <div className={`structure-flow__node structure-flow__node--${node.tone}`}>
              <strong>{node.label}</strong>
              <span>{node.dtype}</span>
            </div>
            <div className="structure-flow__shape">{node.shape}</div>
            <span className="structure-flow__badge">Used by calculator</span>
            {index < nodes.length - 1 ? (
              <span className="structure-flow__connector" aria-hidden="true" />
            ) : null}
          </div>
        ))}
      </div>
      <aside className="structure-flow__bracket">
        <span>calculator path</span>
      </aside>
    </div>
  );
}

function ScheduleBar({
  title,
  items
}: {
  title: string;
  items: { label: string; count: number; tone: string }[];
}) {
  const total = items.reduce((sum, item) => sum + item.count, 0);

  return (
    <article className="panel">
      <h3>{title}</h3>
      <div className="schedule-bar" aria-label={title}>
        {items.map((item) => (
          <div
            key={item.label}
            className={`schedule-bar__segment schedule-bar__segment--${item.tone}`}
            style={{ flexGrow: item.count }}
            title={`${item.label}: ${item.count}`}
          >
            {item.count}
          </div>
        ))}
      </div>
      <div className="schedule-legend">
        {items.map((item) => (
          <span key={item.label}>
            {item.label}: {item.count} / {total}
          </span>
        ))}
      </div>
    </article>
  );
}

function ParameterTable({ model }: { model: ModelDefinition }) {
  const isDense =
    model.architectureKind === "dense-decoder" ||
    model.architectureKind === "dense-decoder-moe";
  const isHybrid =
    model.architectureKind === "hybrid-linear-moe" ||
    model.architectureKind === "hybrid-linear-dense";
  const isHybridDense = model.architectureKind === "hybrid-linear-dense";

  const rows = isHybrid
    ? [
      ["hidden_size", model.hiddenSize, "Hidden dimension", "Prefill FLOPs"],
      ["num_hidden_layers", model.decoderLayers, "Decoder layer count", "Prefill FLOPs"],
      ["num_attention_heads", model.attentionHeads, "Full attention head count", "Prefill / Decode"],
      ["num_key_value_heads", model.kvHeads, "Full attention KV heads (GQA)", "Decode Cache"],
      ["head_dim", model.headDim, "Full attention per-head dimension", "Prefill / Decode"],
      ["full_attention_layers", model.fullAttentionLayerCount ?? 0, "Full (GQA) attention layers", "Prefill FLOPs / Decode Cache"],
      ["linear_attention_layers", model.linearAttentionLayerCount ?? 0, "Gated DeltaNet linear attention layers", "Prefill FLOPs / Decode State"],
      ["linear_num_key_heads", model.linearNumKeyHeads ?? "-", "Linear attn key heads", "Prefill FLOPs"],
      ["linear_key_head_dim", model.linearKeyHeadDim ?? "-", "Linear attn key head dim", "Prefill FLOPs"],
      ["linear_num_value_heads", model.linearNumValueHeads ?? "-", "Linear attn value heads", "Prefill FLOPs"],
      ["linear_value_head_dim", model.linearValueHeadDim ?? "-", "Linear attn value head dim", "Prefill FLOPs"],
      ["linear_conv_kernel_dim", model.linearConvKernelDim ?? "-", "Linear attn conv kernel size", "Prefill FLOPs"],
      [
        isHybridDense ? "intermediate_size" : "moe_intermediate_size",
        isHybridDense ? (model.intermediateSize ?? model.moeIntermediateSize) : model.moeIntermediateSize,
        isHybridDense ? "Dense SwiGLU FFN width" : "Expert FFN width",
        "Prefill FLOPs"
      ],
      ...(
        isHybridDense
          ? []
          : [
              ["num_experts", model.moeExperts, "Total routed experts", "Weight Memory"],
              ["num_experts_per_tok", model.activeExperts, "Active experts per token", "Prefill FLOPs"]
            ]
      ),
      ["totalParamsB", `${model.totalParamsB.toFixed(1)} B`, "Total parameters (billions)", "Weight Memory"],
      ["totalExpertParamsB", `${model.totalExpertParamsB.toFixed(1)} B`, "Expert parameters (billions)", "Weight Memory"],
      ["estimatedWeightsGb", `${model.estimatedWeightsGb.toFixed(2)} GB`, "Static weight estimate (reference)", "Decode Memory"]
    ]
    : isDense
    ? [
      ["hidden_size", model.hiddenSize, "Hidden dimension", "Prefill FLOPs"],
      ["num_hidden_layers", model.decoderLayers, "Decoder layer count", "Prefill FLOPs"],
      ["num_attention_heads", model.attentionHeads, "Attention head count", "Prefill / Decode"],
      ["num_key_value_heads", model.kvHeads, "KV heads (sliding, GQA)", "Decode Cache"],
      ["head_dim", model.headDim, "Per-head dimension (sliding)", "Prefill / Decode"],
      ["global_head_dim", model.globalHeadDim ?? model.headDim, "Full attention head dim", "Prefill FLOPs / Decode Cache"],
      ["num_global_kv_heads", model.numGlobalKeyValueHeads ?? "-", "KV heads (full, MQA)", "Decode Cache"],
      ["sliding_attention_layers", model.slidingAttentionLayerCount ?? model.slidingLayerCount, "Sliding attention layers", "Prefill FLOPs / Decode Cache"],
      ["full_attention_layers", model.fullAttentionLayerCount ?? 0, "Full (global) attention layers", "Prefill FLOPs / Decode Cache"],
      ["sliding_window", model.slidingWindow, "Local visible window size", "Prefill Core / Decode L_kv"],
      ["intermediate_size", model.intermediateSize ?? "-", "FFN intermediate width", "Prefill FLOPs"],
      ["hidden_activation", model.hiddenActivation ?? "gelu_pytorch_tanh", "FFN activation (GeGLU)", "Compute"],
      ["totalParamsB", `${model.totalParamsB.toFixed(1)} B`, "Total parameters (billions)", "Weight Memory"],
      ["totalExpertParamsB", `${model.totalExpertParamsB.toFixed(1)} B`, "Expert parameters (billions)", "Weight Memory"],
      ["estimatedWeightsGb", `${model.estimatedWeightsGb.toFixed(2)} GB`, "Static weight estimate (reference)", "Decode Memory"]
    ]
    : [
      ["hidden_size", model.hiddenSize, "Hidden dimension", "Prefill FLOPs"],
      ["num_hidden_layers", model.decoderLayers, "Decoder layer count", "Prefill FLOPs"],
      ["num_attention_heads", model.attentionHeads, "Attention head count", "Prefill / Decode"],
      ["num_key_value_heads", model.kvHeads, "Shared-KV MQA heads", "Decode Cache"],
      ["head_dim", model.headDim, "Per-head dimension", "Prefill / Decode"],
      ["q_lora_rank", model.qLoraRank, "Q low-rank projection", "Prefill FLOPs"],
      ["o_lora_rank", model.oLoraRank, "Output projection rank", "Prefill FLOPs"],
      ["sliding_layer_count", model.slidingLayerCount, "Number of sliding attention layers", "Prefill FLOPs / Decode Cache"],
      ["csa_layer_count", model.csaLayerCount, "Number of compressed sparse attention layers", "Prefill FLOPs / Decode Cache"],
      ["hca_layer_count", model.hcaLayerCount, "Number of heavily compressed attention layers", "Prefill FLOPs / Decode Cache"],
      ["sliding_window", model.slidingWindow, "Local visible window size", "Prefill Core / Decode L_kv"],
      ["index_topk", model.indexTopk, "CSA selected compressed blocks", "Prefill / Decode"],
      ["compress_rate_csa", model.csaCompressRate, "CSA compression rate", "CSA Cache / Decode L_kv"],
      ["compress_rate_hca", model.hcaCompressRate, "HCA compression rate", "HCA Cache / Decode L_kv"],
      ["moe_intermediate_size", model.moeIntermediateSize, "Expert FFN width", "Prefill FLOPs"],
      ["totalParamsB", `${model.totalParamsB.toFixed(1)} B`, "Total parameters (billions)", "Weight Memory"],
      ["totalExpertParamsB", `${model.totalExpertParamsB.toFixed(1)} B`, "Expert parameters (billions)", "Weight Memory"],
      ["estimatedWeightsGb", `${model.estimatedWeightsGb.toFixed(2)} GB`, "Static weight estimate (reference)", "Decode Memory"]
    ];

  return (
    <article className="panel panel--large">
      <h3>配置参数与性能关联</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th>Field</th>
            <th>Value</th>
            <th>Meaning</th>
            <th>Used In</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([field, value, meaning, usedIn]) => (
            <tr key={field}>
              <td>{field}</td>
              <td>{value}</td>
              <td>{meaning}</td>
              <td>{usedIn}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

export function ModelStructurePage() {
  const {
    structureModelId,
    structureSelectedFamily,
    availableFamilies,
    structureAvailableModels,
    structureSelectedModel: model,
    updateStructureModelFamily,
    updateStructureModelId
  } = useCalculatorContext();

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Architecture View</p>
          <h2>模型结构</h2>
        </div>
        <p className="page-description">
          该页面将展示选中模型的层级结构、关键超参与模块组织。
        </p>
      </div>

      <article className="panel model-hero">
        <div className="model-hero__selector">
          <label className="field">
            <span>模型家族</span>
            <select
              value={structureSelectedFamily}
              onChange={(event) => updateStructureModelFamily(event.target.value)}
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
              value={structureModelId}
              onChange={(event) => updateStructureModelId(event.target.value as ModelId)}
            >
              {structureAvailableModels.map((entry) => (
                <option key={entry.id} value={entry.id}>
                  {entry.displayName}
                </option>
              ))}
            </select>
          </label>
          <div>
            <h3>{model.displayName}</h3>
            <p>
              {model.architectureKind === "hybrid-linear-moe"
                ? "Hybrid decoder-only architecture with Gated DeltaNet linear attention, full GQA anchor layers, and routed MoE."
                : model.architectureKind === "hybrid-linear-dense"
                ? "Hybrid decoder-only architecture with Gated DeltaNet linear attention, full GQA anchor layers, and dense SwiGLU feed-forward blocks."
                : model.architectureKind === "dense-decoder-moe"
                ? "Dense decoder-only architecture with sliding/full attention and routed MoE feed-forward blocks."
                : model.architectureKind === "dense-decoder"
                ? "Dense decoder-only architecture with GQA/MQA attention, sliding+full layer pattern, and GeGLU MLP."
                : "DeepSeek V4 decoder-only architecture with compressed attention, mHC residual streams, and routed MoE."}
            </p>
          </div>
        </div>
        <MetricStrip model={model} />
      </article>

      <div className="structure-grid">
        <article className="panel panel--large">
          <h3>结构流图</h3>
          <StructureFlowDiagram model={model} />
          <div className="structure-note">
            <span>Used by calculator</span>
            <p>
              `decoder layers`, `attention heads`, `head dim`, `MoE width`, and
              compression schedule directly feed the performance formulas.
            </p>
          </div>
        </article>

        <aside className="module-stack">
          {model.architectureKind === "hybrid-linear-moe" ||
          model.architectureKind === "hybrid-linear-dense" ? (
            <>
              <article className="panel">
                <h3>Full Attention (GQA)</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Heads</dt>
                    <dd>{model.attentionHeads}</dd>
                  </div>
                  <div>
                    <dt>KV Heads</dt>
                    <dd>{model.kvHeads}</dd>
                  </div>
                  <div>
                    <dt>Head Dim</dt>
                    <dd>{model.headDim}</dd>
                  </div>
                  <div>
                    <dt>GQA Ratio</dt>
                    <dd>{model.attentionHeads / model.kvHeads}:1</dd>
                  </div>
                  <div>
                    <dt>Q+Gate Proj</dt>
                    <dd>D → 2·n_h·c = {model.hiddenSize * 2 * model.attentionHeads * model.headDim}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>Linear Attention (Gated DeltaNet)</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Key Heads / Dim</dt>
                    <dd>{model.linearNumKeyHeads} / {model.linearKeyHeadDim}</dd>
                  </div>
                  <div>
                    <dt>Value Heads / Dim</dt>
                    <dd>{model.linearNumValueHeads} / {model.linearValueHeadDim}</dd>
                  </div>
                  <div>
                    <dt>Conv Kernel</dt>
                    <dd>{model.linearConvKernelDim}</dd>
                  </div>
                  <div>
                    <dt>Layers</dt>
                    <dd>{model.linearAttentionLayerCount}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>{model.architectureKind === "hybrid-linear-dense" ? "Dense SwiGLU FFN" : "MoE"}</h3>
                <dl className="summary-list summary-list--compact">
                  {model.architectureKind === "hybrid-linear-moe" ? (
                    <>
                      <div>
                        <dt>Routed Experts</dt>
                        <dd>{model.moeExperts}</dd>
                      </div>
                      <div>
                        <dt>Active / Token</dt>
                        <dd>{model.activeExperts}</dd>
                      </div>
                    </>
                  ) : null}
                  <div>
                    <dt>Intermediate</dt>
                    <dd>{model.intermediateSize ?? model.moeIntermediateSize}</dd>
                  </div>
                  {model.architectureKind === "hybrid-linear-moe" ? (
                    <div>
                      <dt>Shared Expert</dt>
                      <dd>Yes (gated)</dd>
                    </div>
                  ) : (
                    <div>
                      <dt>Routing</dt>
                      <dd>None (dense)</dd>
                    </div>
                  )}
                </dl>
              </article>
            </>
          ) : model.architectureKind === "dense-decoder" || model.architectureKind === "dense-decoder-moe" ? (
            <>
              <article className="panel">
                <h3>Attention</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Heads</dt>
                    <dd>{model.attentionHeads}</dd>
                  </div>
                  <div>
                    <dt>KV Heads (sliding)</dt>
                    <dd>{model.kvHeads}</dd>
                  </div>
                  <div>
                    <dt>Head Dim (sliding)</dt>
                    <dd>{model.headDim}</dd>
                  </div>
                  <div>
                    <dt>KV Heads (full)</dt>
                    <dd>{model.numGlobalKeyValueHeads ?? "-"}</dd>
                  </div>
                  <div>
                    <dt>Head Dim (full)</dt>
                    <dd>{model.globalHeadDim ?? "-"}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>Sliding / Full Pattern</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Sliding Window</dt>
                    <dd>{model.slidingWindow}</dd>
                  </div>
                  <div>
                    <dt>Sliding Layers</dt>
                    <dd>{model.slidingAttentionLayerCount ?? model.slidingLayerCount}</dd>
                  </div>
                  <div>
                    <dt>Full Layers</dt>
                    <dd>{model.fullAttentionLayerCount ?? 0}</dd>
                  </div>
                  <div>
                    <dt>K=V Shared (full)</dt>
                    <dd>{model.attentionKEqV ? "Yes" : "No"}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>{model.architectureKind === "dense-decoder-moe" ? "MoE FFN" : "FFN"}</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Type</dt>
                    <dd>{model.architectureKind === "dense-decoder-moe" ? "Routed MoE" : "GeGLU"}</dd>
                  </div>
                  <div>
                    <dt>Intermediate</dt>
                    <dd>
                      {model.architectureKind === "dense-decoder-moe"
                        ? model.moeIntermediateSize
                        : model.intermediateSize ?? "-"}
                    </dd>
                  </div>
                  {model.architectureKind === "dense-decoder-moe" ? (
                    <div>
                      <dt>Experts</dt>
                      <dd>{model.activeExperts} / {model.moeExperts}</dd>
                    </div>
                  ) : null}
                  <div>
                    <dt>Activation</dt>
                    <dd>{model.hiddenActivation ?? "gelu_pytorch_tanh"}</dd>
                  </div>
                  <div>
                    <dt>Layers</dt>
                    <dd>All {model.decoderLayers}</dd>
                  </div>
                </dl>
              </article>
            </>
          ) : (
            <>
              <article className="panel">
                <h3>Attention</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Heads</dt>
                    <dd>{model.attentionHeads}</dd>
                  </div>
                  <div>
                    <dt>KV Heads</dt>
                    <dd>{model.kvHeads}</dd>
                  </div>
                  <div>
                    <dt>Head Dim</dt>
                    <dd>{model.headDim}</dd>
                  </div>
                  <div>
                    <dt>Q Rank</dt>
                    <dd>{model.qLoraRank}</dd>
                  </div>
                  <div>
                    <dt>O Groups</dt>
                    <dd>{model.oGroups}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>Compressed Cache</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Sliding Window</dt>
                    <dd>{model.slidingWindow}</dd>
                  </div>
                  <div>
                    <dt>CSA TopK</dt>
                    <dd>{model.indexTopk}</dd>
                  </div>
                  <div>
                    <dt>CSA Rate</dt>
                    <dd>{model.csaCompressRate}</dd>
                  </div>
                  <div>
                    <dt>HCA Rate</dt>
                    <dd>{model.hcaCompressRate}</dd>
                  </div>
                </dl>
              </article>

              <article className="panel">
                <h3>MoE</h3>
                <dl className="summary-list summary-list--compact">
                  <div>
                    <dt>Routed Experts</dt>
                    <dd>{model.moeExperts}</dd>
                  </div>
                  <div>
                    <dt>Active / Token</dt>
                    <dd>{model.activeExperts}</dd>
                  </div>
                  <div>
                    <dt>Intermediate</dt>
                    <dd>{model.moeIntermediateSize}</dd>
                  </div>
                </dl>
              </article>
            </>
          )}
        </aside>
      </div>

      <div className="panel-grid">
        {model.architectureKind === "hybrid-linear-moe" ||
        model.architectureKind === "hybrid-linear-dense" ? (
          <>
            <ScheduleBar
              title="Attention Schedule"
              items={[
                { label: "Linear (Gated DeltaNet)", count: model.linearAttentionLayerCount ?? 0, tone: "teal" },
                { label: "Full (GQA)", count: model.fullAttentionLayerCount ?? 0, tone: "blue" }
              ]}
            />
            <ScheduleBar
              title={model.architectureKind === "hybrid-linear-dense" ? "MLP Schedule" : "MoE Schedule"}
              items={[
                {
                  label:
                    model.architectureKind === "hybrid-linear-dense"
                      ? "Dense SwiGLU"
                      : "MoE (routed 256 + shared)",
                  count: model.decoderLayers,
                  tone: "green"
                }
              ]}
            />
          </>
        ) : model.architectureKind === "dense-decoder" || model.architectureKind === "dense-decoder-moe" ? (
          <>
            <ScheduleBar
              title="Attention Schedule"
              items={[
                { label: "Sliding", count: model.slidingAttentionLayerCount ?? model.slidingLayerCount, tone: "neutral" },
                { label: "Full", count: model.fullAttentionLayerCount ?? 0, tone: "blue" }
              ]}
            />
            <ScheduleBar
              title={model.architectureKind === "dense-decoder-moe" ? "MoE Schedule" : "MLP Schedule"}
              items={[
                {
                  label: model.architectureKind === "dense-decoder-moe" ? "Routed MoE" : "Dense GeGLU",
                  count: model.decoderLayers,
                  tone: "green"
                }
              ]}
            />
          </>
        ) : (
          <>
            <ScheduleBar
              title="Attention Schedule"
              items={[
                { label: "Sliding", count: model.slidingLayerCount, tone: "neutral" },
                { label: "CSA", count: model.csaLayerCount, tone: "blue" },
                { label: "HCA", count: model.hcaLayerCount, tone: "teal" }
              ]}
            />
            <ScheduleBar
              title="MLP Schedule"
              items={[
                { label: "Hash MoE", count: Math.min(3, model.decoderLayers), tone: "amber" },
                { label: "MoE", count: Math.max(model.decoderLayers - 3, 0), tone: "green" }
              ]}
            />
          </>
        )}
      </div>

      <ParameterTable model={model} />
    </section>
  );
}
