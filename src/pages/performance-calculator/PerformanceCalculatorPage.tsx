import { ComparisonTable } from "../../features/performance-calculator/components/ComparisonTable";
import { CalculatorControls } from "../../features/performance-calculator/components/CalculatorControls";
import { FormulaTraceCard } from "../../features/performance-calculator/components/FormulaTraceCard";
import { IntermediateMetricsTable } from "../../features/performance-calculator/components/IntermediateMetricsTable";
import { MemoryBreakdownCard } from "../../features/performance-calculator/components/MemoryBreakdownCard";
import { MetricCards } from "../../features/performance-calculator/components/MetricCards";
import { TrendChart } from "../../features/performance-calculator/components/TrendChart";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

export function PerformanceCalculatorPage() {
  const {
    state,
    result,
    status,
    selectedModel,
    selectedFamily,
    availableFamilies,
    availableModels,
    validationErrors,
    updateModelFamily,
    updateModelId,
    updatePlatform,
    updateWorkload,
    updateView,
    applyQuickRange,
    reset,
    calculate
  } = useCalculatorContext();

  if (!result) {
    return null;
  }

  const statusText =
    status === "invalid"
      ? "参数存在校验错误"
      : status === "calculating"
        ? "计算中"
        : status === "calculated"
          ? "结果已更新"
          : "待计算";

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Primary Workspace</p>
          <h2>性能计算</h2>
        </div>
        <p className="page-description">
          输入模型、平台参数和 token 范围，计算 prefill / decode 性能并查看趋势。
        </p>
      </div>

      <CalculatorControls
        modelId={state.modelId}
        selectedFamily={selectedFamily}
        availableFamilies={availableFamilies}
        availableModels={availableModels}
        selectedModel={selectedModel}
        platform={state.platform}
        workload={state.workload}
        view={state.view}
        validationErrors={validationErrors}
        onModelFamilyChange={updateModelFamily}
        onModelIdChange={updateModelId}
        onPlatformChange={updatePlatform}
        onWorkloadChange={updateWorkload}
        onViewChange={updateView}
        onQuickRange={applyQuickRange}
      />

      <div className="performance-primary-results">
        <div className="toolbar">
          <div className="toolbar__actions">
            <button type="button" className="primary-button" onClick={calculate}>
              计算性能
            </button>
            <button type="button" className="secondary-button" onClick={reset}>
              重置
            </button>
          </div>
          <p className="status-pill">{statusText}</p>
        </div>

        <div className="performance-overview-row">
          <div className="performance-overview-row__metrics">
            <MetricCards summary={result.summary} />
          </div>
          <MemoryBreakdownCard
            rows={result.memoryBreakdown}
            weightSourceUrl={selectedModel.weightSourceUrl}
            runtimeOverheadGb={state.platform.runtimeOverheadGb}
            runtimeOverheadError={validationErrors.runtimeOverheadGb}
            onRuntimeOverheadChange={(value) => updatePlatform("runtimeOverheadGb", value)}
          />
        </div>
      </div>

      <div className="performance-results">
        <ComparisonTable rows={result.comparisonRows} />

        <div className="performance-summary-row">
          <article className="panel panel--large">
            <h3>结构摘要</h3>
            <p>显示模型层数、hidden size、MoE 配置和上下文长度。</p>
            <dl className="summary-list summary-list--compact">
              <div>
                <dt>Decoder Layers</dt>
                <dd>{selectedModel.decoderLayers}</dd>
              </div>
              <div>
                <dt>Hidden Size</dt>
                <dd>{selectedModel.hiddenSize.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Attention Heads</dt>
                <dd>{selectedModel.attentionHeads}</dd>
              </div>
              <div>
                <dt>KV Heads</dt>
                <dd>{selectedModel.kvHeads}</dd>
              </div>
              <div>
                <dt>Experts</dt>
                <dd>{selectedModel.moeExperts}</dd>
              </div>
              <div>
                <dt>Active Experts / Token</dt>
                <dd>{selectedModel.activeExperts}</dd>
              </div>
              <div>
                <dt>MoE Intermediate Size</dt>
                <dd>{selectedModel.moeIntermediateSize.toLocaleString()}</dd>
              </div>
              <div>
                <dt>Context Limit</dt>
                <dd>{selectedModel.contextLimit.toLocaleString()}</dd>
              </div>
            </dl>
          </article>

            <article className="panel panel--large">
              <h3>当前上下文摘要</h3>
              <div className="stack-list">
                <div className="stack-list__row">
                  <strong>Model</strong>
                  <span>{selectedModel.displayName}</span>
                </div>
                <div className="stack-list__row">
                  <strong>Prompt Length</strong>
                  <span>{state.workload.prefillTokenLength.toLocaleString()} tokens</span>
                </div>
                <div className="stack-list__row">
                  <strong>Initial Decode Context</strong>
                  <span>{state.workload.prefillTokenLength.toLocaleString()} tokens (from prompt KV cache)</span>
                </div>
                <div className="stack-list__row">
                  <strong>Bytes / Weight</strong>
                  <span>{state.platform.bytesPerWeight} B/param</span>
                </div>
                <div className="stack-list__row">
                  <strong>Bytes / Activation</strong>
                  <span>{state.platform.bytesPerActivation} B/elem</span>
                </div>
                <div className="stack-list__row">
                  <strong>Bytes / Expert</strong>
                  <span>{state.platform.bytesPerExpert} B/param</span>
                </div>
              </div>
            </article>
        </div>

        <TrendChart
          points={result.tokenSweepSeries}
          selectedMetric={state.view.selectedTrendMetric}
          onMetricChange={(value) => updateView("selectedTrendMetric", value)}
          showDataPoints={state.view.showTrendDataPoints}
          onShowDataPointsChange={(value) => updateView("showTrendDataPoints", value)}
          showBottleneckBackground={state.view.showBottleneckBackground}
          onShowBottleneckBackgroundChange={(value) =>
            updateView("showBottleneckBackground", value)
          }
        />

        {state.view.showFormulaTrace ? (
          <FormulaTraceCard sections={result.formulaTrace} />
        ) : null}

        {state.view.showIntermediateMetrics ? (
          <IntermediateMetricsTable rows={result.intermediateMetrics} />
        ) : null}
      </div>
    </section>
  );
}
