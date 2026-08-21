import type { WorkloadInput } from "../../../domain/workload/types";
import { getModelDefinition } from "../../../engines/model-registry";
import type { ComparisonProfile, ComparisonResult } from "../types/comparison";
import { calculatePerformanceResult } from "./performanceCalculator";

export function calculateComparisonResults(
  profiles: ComparisonProfile[],
  workload: WorkloadInput
): ComparisonResult[] {
  return profiles.filter((profile) => profile.enabled).map((profile) => {
    const model = getModelDefinition(profile.modelId);
    if (workload.tokenRangeEnd > model.contextLimit) {
      throw new Error(`${profile.label} 的 Token Sweep End 超过模型上下文上限 ${model.contextLimit.toLocaleString()}`);
    }
    let result;
    try {
      result = calculatePerformanceResult(model, profile.platform, workload);
    } catch (error) {
      throw new Error(`${profile.label} 计算失败：${error instanceof Error ? error.message : "未知错误"}`);
    }

    return {
      profile,
      tokenSweepSeries: result.tokenSweepSeries,
      logarithmicTokenSweepSeries: result.logarithmicTokenSweepSeries,
      memorySweepSeries: result.tokenSweepSeries.map((point) => ({
        tokenLength: point.tokenLength,
        weightsGb: point.weightsGb,
        persistentCacheGb: point.persistentCacheGb,
        temporaryMemoryGb: point.temporaryMemoryGb,
        runtimeOverheadGb: point.runtimeOverheadGb,
        totalGb: point.totalRuntimeMemoryGb
      }))
    };
  });
}
