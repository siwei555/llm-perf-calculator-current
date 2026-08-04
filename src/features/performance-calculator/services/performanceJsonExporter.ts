import type { ModelDefinition } from "../../../domain/model/types";
import type { PerformanceResult } from "../../../domain/performance/types";
import type { CalculationSnapshot } from "../state/useCalculatorState";

export type PerformanceJsonExportInput = {
  model: ModelDefinition;
  snapshot: CalculationSnapshot;
  result: PerformanceResult;
  exportedAt?: Date;
};

function safeFilePart(value: string) {
  return value.replace(/[\\/:*?"<>|]+/g, "-").replace(/\s+/g, "-");
}

export function buildPerformanceJson({
  model,
  snapshot,
  result,
  exportedAt = new Date()
}: PerformanceJsonExportInput) {
  return {
    schemaVersion: 1,
    metadata: {
      exportedAt: exportedAt.toISOString(),
      generator: "LLM Perf Calculator",
      modelId: model.id,
      modelName: model.displayName,
      modelFamily: model.family,
      formulaStrategy: model.formulaStrategyId,
      note: "Engineering estimate; values are not benchmark measurements."
    },
    units: {
      computeThroughputTflops: "TFLOPS",
      memoryBandwidthGbps: "GB/s",
      memoryCapacityGb: "GB",
      runtimeOverheadGb: "GB",
      tokenLength: "tokens",
      ttftMs: "ms",
      tps: "tokens/s",
      memoryGb: "GB",
      gflopsPerToken: "GFLOPs/token"
    },
    model,
    platform: snapshot.platform,
    workload: snapshot.workload,
    results: {
      summary: result.summary,
      comparison: result.comparisonRows,
      memoryBreakdown: result.memoryBreakdown
    },
    prefillProjection: result.projectionSeries.map((point) => ({
      contextLength: point.contextLength,
      gflopsPerToken: point.prefillGflopsPerToken,
      estimatedTpsAt20PercentEfficiency: point.prefillTps20,
      estimatedTpsAt40PercentEfficiency: point.prefillTps40,
      estimatedTtftSecondsAt40PercentEfficiency: point.prefillTtftSec40
    })),
    decodeProjection: result.projectionSeries.map((point) => ({
      contextLength: point.contextLength,
      persistentCacheGb: point.persistentCacheGb,
      temporaryMemoryGb: point.temporaryMemoryGb,
      totalMemoryGb: point.totalMemoryGb,
      estimatedTpsAt40PercentEfficiency: point.decodeTps40,
      estimatedTpsAt60PercentEfficiency: point.decodeTps60,
      estimatedTpsAt80PercentEfficiency: point.decodeTps80
    })),
    intermediateMetrics: result.intermediateMetrics,
    formulaTrace: result.formulaTrace,
    tokenTrend: result.tokenSweepSeries
  };
}

export function exportPerformanceJson(input: PerformanceJsonExportInput) {
  const exportedAt = input.exportedAt ?? new Date();
  const payload = buildPerformanceJson({ ...input, exportedAt });
  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: "application/json;charset=utf-8"
  });
  const date = exportedAt.toISOString().slice(0, 10);
  const filename = `llm-perf-${safeFilePart(input.model.displayName)}-${date}.json`;
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  return filename;
}
