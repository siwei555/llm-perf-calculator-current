export type BottleneckType =
  | "compute-bound"
  | "bandwidth-bound";

export type TrendMetricKey =
  | "prefillTps"
  | "decodeTps"
  | "ttftMs"
  | "totalRuntimeMemoryGb";

export type CalculationStatus =
  | "idle"
  | "invalid"
  | "ready"
  | "calculating"
  | "calculated";

export type PerformanceSummary = {
  ttftMs: number;
  prefillTps: number;
  initialDecodeTps: number;
  averageDecodeTps: number;
  decodeTimeMs: number;
  finalDecodeContext: number;
  peakRuntimeMemoryGb: number;
  /** Backward-compatible alias of averageDecodeTps. */
  decodeTps: number;
  totalRuntimeMemoryGb: number;
  prefillBottleneck: BottleneckType;
  decodeBottleneck: BottleneckType;
  memoryFitsCapacity: boolean;
};

export type ComparisonRow = {
  label: string;
  unit: string;
  prefill: string;
  decode: string;
};

export type MemoryBreakdownRow = {
  key: "weights" | "persistentDecodeCache" | "peakTempWorkingSet" | "runtimeOverhead" | "estimatedTotal";
  label: string;
  valueGb: number;
};

export type IntermediateMetric = {
  key: string;
  label: string;
  symbol: string;
  value: string;
  unit: string;
  source: "config" | "derived" | "formula";
};

export type FormulaTraceRow = {
  label: string;
  expression: string;
  /** 对公式口径的自然语言说明；与纯数学表达式分开渲染。 */
  explanation?: string[];
  evaluated: string;
  sourceLabel?: string;
  sourceUrl?: string;
};

export type FormulaTraceSection = {
  category: "prefill" | "decode" | "memory";
  rows: FormulaTraceRow[];
};

export type TokenSweepPoint = {
  tokenLength: number;
  prefillTps: number;
  decodeTps: number;
  ttftMs: number;
  totalRuntimeMemoryGb: number;
  weightsGb: number;
  persistentCacheGb: number;
  temporaryMemoryGb: number;
  runtimeOverheadGb: number;
  prefillBottleneck: BottleneckType;
  decodeBottleneck: BottleneckType;
};

export type MemorySweepPoint = {
  tokenLength: number;
  weightsGb: number;
  persistentCacheGb: number;
  temporaryMemoryGb: number;
  runtimeOverheadGb: number;
  totalGb: number;
};

export type PerformanceProjectionPoint = {
  contextLength: number;
  prefillGflopsPerToken: number;
  prefillTps20: number;
  prefillTps40: number;
  prefillTtftSec40: number;
  persistentCacheGb: number;
  temporaryMemoryGb: number;
  totalMemoryGb: number;
  decodeTps40: number;
  decodeTps60: number;
  decodeTps80: number;
};

export type PerformanceResult = {
  summary: PerformanceSummary;
  comparisonRows: ComparisonRow[];
  memoryBreakdown: MemoryBreakdownRow[];
  intermediateMetrics: IntermediateMetric[];
  formulaTrace: FormulaTraceSection[];
  tokenSweepSeries: TokenSweepPoint[];
  logarithmicTokenSweepSeries: TokenSweepPoint[];
  projectionSeries: PerformanceProjectionPoint[];
};
