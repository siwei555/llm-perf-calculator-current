import { useState } from "react";
import type { ModelDefinition, ModelId } from "../../../domain/model/types";
import type { PlatformInput } from "../../../domain/platform/types";
import type { WorkloadInput } from "../../../domain/workload/types";
import type { ModelFamilyOption } from "../../../engines/model-registry";
import type {
  CalculatorValidation,
  CalculatorViewState,
  QuickRangeTarget
} from "../state/useCalculatorState";

type Props = {
  modelId: ModelId;
  selectedFamily: string;
  availableFamilies: ModelFamilyOption[];
  availableModels: ModelDefinition[];
  selectedModel: ModelDefinition;
  platform: PlatformInput;
  workload: WorkloadInput;
  view: CalculatorViewState;
  validationErrors: CalculatorValidation;
  onModelFamilyChange: (family: string) => void;
  onModelIdChange: (modelId: ModelId) => void;
  onPlatformChange: <K extends keyof PlatformInput>(key: K, value: PlatformInput[K]) => void;
  onWorkloadChange: <K extends keyof WorkloadInput>(key: K, value: WorkloadInput[K]) => void;
  onViewChange: <K extends keyof CalculatorViewState>(
    key: K,
    value: CalculatorViewState[K]
  ) => void;
  onQuickRange: (target: QuickRangeTarget, tokenLength: number) => void;
};

function numberValue(value: string) {
  return Number(value);
}

function optionalNumberValue(value: string) {
  return value === "" ? null : Number(value);
}

function FieldError({ message }: { message?: string }) {
  if (!message) {
    return null;
  }

  return <span className="field-error">{message}</span>;
}

const quickInputLabels: Record<QuickRangeTarget, string> = {
  prefillTokenLength: "Prompt Token Length",
  decodeOutputTokens: "Decode Output Tokens",
  tokenRangeStart: "Token Sweep Start",
  tokenRangeEnd: "Token Sweep End",
  tokenRangeStep: "Token Sweep Step"
};

export function CalculatorControls({
  modelId,
  selectedFamily,
  availableFamilies,
  availableModels,
  selectedModel,
  platform,
  workload,
  view,
  validationErrors,
  onModelFamilyChange,
  onModelIdChange,
  onPlatformChange,
  onWorkloadChange,
  onViewChange,
  onQuickRange
}: Props) {
  const [quickInputTarget, setQuickInputTarget] =
    useState<QuickRangeTarget>("prefillTokenLength");

  return (
    <div className="calculator-controls">
      <div className="panel-grid panel-grid--controls">
        <article id="performance-model-selection" className="panel panel--model-selection page-section-anchor">
          <h3>模型选择</h3>
          <label className="field">
            <span>模型家族</span>
            <select
              value={selectedFamily}
              onChange={(event) => onModelFamilyChange(event.target.value)}
            >
              {availableFamilies.map((family) => (
                <option key={family.id} value={family.id}>
                  {family.displayName}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>模型</span>
            <select
              value={modelId}
              onChange={(event) => onModelIdChange(event.target.value as ModelId)}
            >
              {availableModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.displayName}
                </option>
              ))}
            </select>
          </label>
          <dl className="summary-list summary-list--model-selection">
            <div>
              <dt>Layers</dt>
              <dd>{selectedModel.decoderLayers}</dd>
            </div>
            <div>
              <dt>Hidden Size</dt>
              <dd>{selectedModel.hiddenSize}</dd>
            </div>
            <div>
              <dt>Active Experts / Token</dt>
              <dd>{selectedModel.activeExperts}</dd>
            </div>
            <div>
              <dt>Routed Experts / Layer</dt>
              <dd>{selectedModel.moeExperts}</dd>
            </div>
            <div>
              <dt>Context</dt>
              <dd>{(selectedModel.contextLimit / 1024).toFixed(0)}K</dd>
            </div>
            {selectedModel.recommendedPrecision && (
              <div>
                <dt>Recommended Precision</dt>
                <dd>{selectedModel.recommendedPrecision.label}</dd>
              </div>
            )}
          </dl>
          {selectedModel.parameterSourceUrl ? (
            <a
              className="model-parameter-source"
              href={selectedModel.parameterSourceUrl}
              target="_blank"
              rel="noreferrer"
            >
              模型参数来源 ↗
            </a>
          ) : null}
        </article>

        <article id="performance-input-length" className="panel page-section-anchor">
          <h3>输入长度</h3>
          <div className="form-grid">
            <label className="field">
              <span>Prompt Token Length</span>
              <input
                type="number"
                value={workload.prefillTokenLength}
                onFocus={() => setQuickInputTarget("prefillTokenLength")}
                onChange={(event) =>
                  onWorkloadChange("prefillTokenLength", numberValue(event.target.value))
                }
              />
              <small>
                默认 131072（128K）是验收计算长度，不是模型最大上下文窗口。
              </small>
              <FieldError message={validationErrors.prefillTokenLength} />
            </label>
            <label className="field">
              <span>Decode Output Tokens</span>
              <input
                type="number"
                value={workload.decodeOutputTokens ?? ""}
                placeholder="Defaults to prompt length"
                onFocus={() => setQuickInputTarget("decodeOutputTokens")}
                onChange={(event) =>
                  onWorkloadChange("decodeOutputTokens", optionalNumberValue(event.target.value))
                }
              />
              <small>留空时按 Prompt Token Length 计算</small>
              <FieldError message={validationErrors.decodeOutputTokens} />
            </label>
          </div>
          <h4 className="input-section-title">Token趋势图扫描</h4>
          <div className="form-grid">
            <label className="field">
              <span>Token Sweep Start</span>
              <input
                type="number"
                value={workload.tokenRangeStart}
                onFocus={() => setQuickInputTarget("tokenRangeStart")}
                onChange={(event) =>
                  onWorkloadChange("tokenRangeStart", numberValue(event.target.value))
                }
              />
              <FieldError message={validationErrors.tokenRangeStart} />
            </label>
            <label className="field">
              <span>Token Sweep End</span>
              <input
                type="number"
                value={workload.tokenRangeEnd}
                onFocus={() => setQuickInputTarget("tokenRangeEnd")}
                onChange={(event) =>
                  onWorkloadChange("tokenRangeEnd", numberValue(event.target.value))
                }
              />
              <FieldError message={validationErrors.tokenRangeEnd} />
            </label>
            <label className="field">
              <span>Token Sweep Step</span>
              <input
                type="number"
                value={workload.tokenRangeStep}
                onFocus={() => setQuickInputTarget("tokenRangeStep")}
                onChange={(event) =>
                  onWorkloadChange("tokenRangeStep", numberValue(event.target.value))
                }
              />
              <FieldError message={validationErrors.tokenRangeStep} />
            </label>
          </div>
          <div className="quick-input-heading">
            <h4 className="input-section-title input-section-title--quick">快捷输入</h4>
            <small>点击目标输入框即可切换快捷输入目标</small>
          </div>
          <p className="quick-input-target">
            当前输入目标：<strong>{quickInputLabels[quickInputTarget]}</strong>
          </p>
          <div className="quick-actions">
            {[4096, 8192, 32768, 65536, 131072, 1048576].map((tokenLength) => {
              const exceedsContext =
                quickInputTarget !== "tokenRangeStep" &&
                tokenLength > selectedModel.contextLimit;

              return (
                <button
                  key={tokenLength}
                  type="button"
                  className="ghost-button"
                  disabled={exceedsContext}
                  title={
                    exceedsContext
                      ? `超过当前模型最大上下文 ${selectedModel.contextLimit.toLocaleString()} tokens`
                      : `填入 ${quickInputLabels[quickInputTarget]}`
                  }
                  onClick={() => onQuickRange(quickInputTarget, tokenLength)}
                >
                  {tokenLength >= 1048576 ? "1M" : `${tokenLength / 1024}K`}
                </button>
              );
            })}
          </div>
        </article>

        <article id="performance-platform" className="panel page-section-anchor">
          <h3>平台参数</h3>
          <div className="form-grid">
            <label className="field">
              <span>Compute Throughput (TFLOPS)</span>
              <input
                type="number"
                value={platform.computeThroughputTflops}
                onChange={(event) =>
                  onPlatformChange("computeThroughputTflops", numberValue(event.target.value))
                }
              />
              <small>FP8 建议：248 TFLOPS</small>
              <FieldError message={validationErrors.computeThroughputTflops} />
            </label>
            <label className="field">
              <span>Memory Bandwidth (GB/s)</span>
              <input
                type="number"
                step="1"
                value={platform.memoryBandwidthGbps}
                onChange={(event) =>
                  onPlatformChange("memoryBandwidthGbps", numberValue(event.target.value))
                }
              />
              <FieldError message={validationErrors.memoryBandwidthGbps} />
            </label>
            <label className="field">
              <span>HBM / VRAM Capacity (GB)</span>
              <input
                type="number"
                value={platform.memoryCapacityGb}
                onChange={(event) =>
                  onPlatformChange("memoryCapacityGb", numberValue(event.target.value))
                }
              />
              <FieldError message={validationErrors.memoryCapacityGb} />
            </label>
            <label className="field">
              <span>Bytes / Weight</span>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="4"
                value={platform.bytesPerWeight}
                onChange={(event) =>
                  onPlatformChange("bytesPerWeight", numberValue(event.target.value))
                }
              />
              <small>1=FP8, 2=BF16</small>
            </label>
            <label className="field">
              <span>Bytes / Activation</span>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="4"
                value={platform.bytesPerActivation}
                onChange={(event) =>
                  onPlatformChange("bytesPerActivation", numberValue(event.target.value))
                }
              />
              <small>Cache element precision</small>
            </label>
            <label className="field">
              <span>Bytes / Expert</span>
              <input
                type="number"
                step="0.5"
                min="0.5"
                max="4"
                value={platform.bytesPerExpert}
                onChange={(event) =>
                  onPlatformChange("bytesPerExpert", numberValue(event.target.value))
                }
              />
              <small>0.5=FP4, 1=FP8</small>
            </label>
          </div>
        </article>

        <article id="performance-assumptions" className="panel page-section-anchor">
          <h3>计算假设</h3>
          <div className="form-grid">
            <label className="field">
              <span>Batch Size</span>
              <input
                type="number"
                value={platform.batchSize}
                onChange={(event) =>
                  onPlatformChange("batchSize", numberValue(event.target.value))
                }
              />
            </label>
            <label className="field">
              <span>Compute Efficiency</span>
              <input
                type="number"
                step="0.01"
                value={platform.computeEfficiency}
                onChange={(event) =>
                  onPlatformChange("computeEfficiency", numberValue(event.target.value))
                }
              />
            </label>
            <label className="field">
              <span>Bandwidth Efficiency</span>
              <input
                type="number"
                step="0.01"
                value={platform.bandwidthEfficiency}
                onChange={(event) =>
                  onPlatformChange("bandwidthEfficiency", numberValue(event.target.value))
                }
              />
            </label>
            <label className="field">
              <span>Prefill Cache Traffic Factor</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={platform.prefillCacheTrafficFactor}
                onChange={(event) =>
                  onPlatformChange("prefillCacheTrafficFactor", numberValue(event.target.value))
                }
              />
              <small>默认 0.10；允许范围 0–1。用于估算 Prefill cache 流量占持久 cache 容量的比例。</small>
              <FieldError message={validationErrors.prefillCacheTrafficFactor} />
            </label>
            <label className="field field--checkbox">
              <input
                type="checkbox"
                checked={view.showIntermediateMetrics}
                onChange={(event) =>
                  onViewChange("showIntermediateMetrics", event.target.checked)
                }
              />
              <span className="field--checkbox__content">
                <span>Show Intermediate Metrics</span>
                <small>
                  显示 FLOPs、带宽、cache、权重显存等中间量，便于核对计算来源。
                </small>
              </span>
            </label>
            <label className="field field--checkbox">
              <input
                type="checkbox"
                checked={view.showFormulaTrace}
                onChange={(event) => onViewChange("showFormulaTrace", event.target.checked)}
              />
              <span className="field--checkbox__content">
                <span>Show Formula Trace</span>
                <small>
                  显示本次结果使用的公式分解和代入值，便于追溯 Prefill、Decode 与内存计算。
                </small>
              </span>
            </label>
          </div>
        </article>
      </div>

    </div>
  );
}
