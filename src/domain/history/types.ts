import type { ModelId } from "../model/types";
import type { PerformanceSummary } from "../performance/types";
import type { PlatformInput } from "../platform/types";
import type { WorkloadInput } from "../workload/types";

export type CalculationHistoryRecord = {
  id: string;
  createdAt: string;
  modelId: ModelId;
  modelFamily: string;
  modelDisplayName: string;
  platform: PlatformInput;
  workload: WorkloadInput;
  result: PerformanceSummary;
};

export type HistoryTimeOrder = "newest" | "oldest";
