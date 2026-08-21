import { useMemo, useState } from "react";
import type { CalculationHistoryRecord } from "../../../domain/history/types";
import type { ModelId } from "../../../domain/model/types";
import type {
  CalculationStatus,
  PerformanceResult,
  TrendMetricKey
} from "../../../domain/performance/types";
import type { PlatformInput } from "../../../domain/platform/types";
import type { WorkloadInput } from "../../../domain/workload/types";
import {
  getModelDefinition,
  getModelFamilies,
  getModelsByFamily
} from "../../../engines/model-registry";
import { calculatePerformanceResult } from "../services/performanceCalculator";
import { calculateComparisonResults } from "../services/comparisonCalculator";
import { defaultParallelEfficiencies } from "../services/platformScaling";
import type {
  ComparisonProfile,
  ComparisonResult,
  PrecisionPresetId
} from "../types/comparison";
import {
  clearStoredHistoryRecords,
  loadHistoryRecords,
  saveHistoryRecords
} from "../../history/services/historyStorage";

export type CalculatorViewState = {
  selectedTrendMetric: TrendMetricKey;
  showFormulaTrace: boolean;
  showIntermediateMetrics: boolean;
  showBottleneckBackground: boolean;
  showTrendDataPoints: boolean;
};

export type CalculatorState = {
  modelId: ModelId;
  platform: PlatformInput;
  precisionPreset: PrecisionPresetId;
  workload: WorkloadInput;
  view: CalculatorViewState;
};

export type CalculatorValidation = Record<string, string>;
export type CalculationSnapshot = {
  modelId: ModelId;
  platform: PlatformInput;
  workload: WorkloadInput;
};

export type QuickRangeTarget =
  | "prefillTokenLength"
  | "decodeOutputTokens"
  | "tokenRangeStart"
  | "tokenRangeEnd"
  | "tokenRangeStep";

const BF16_COMPUTE_THROUGHPUT_TFLOPS = 124;
const FP8_COMPUTE_THROUGHPUT_TFLOPS = 248;

function defaultComputeThroughputTflops(modelId: ModelId) {
  const precisionLabel = getModelDefinition(modelId).recommendedPrecision.label;
  return /FP8/i.test(precisionLabel)
    ? FP8_COMPUTE_THROUGHPUT_TFLOPS
    : BF16_COMPUTE_THROUGHPUT_TFLOPS;
}

const defaultState: CalculatorState = {
  modelId: "deepseek-v4-flash",
  precisionPreset: "custom",
  platform: {
    computeThroughputTflops: defaultComputeThroughputTflops("deepseek-v4-flash"),
    memoryBandwidthGbps: 273,
    memoryCapacityGb: 128,
    computeEfficiency: 0.4,
    bandwidthEfficiency: 0.6,
    prefillCacheTrafficFactor: 0.1,
    batchSize: 1,
    chipCount: 1,
    ...defaultParallelEfficiencies(1),
    runtimeOverheadGb: 4,
    bytesPerWeight: 1,
    bytesPerActivation: 2,
    bytesPerExpert: 0.5
  },
  workload: {
    prefillTokenLength: 131072,
    decodeOutputTokens: null,
    tokenRangeStart: 4096,
    tokenRangeEnd: 131072,
    tokenRangeStep: 4096,
    logarithmicScaleMultiplier: 2,
    tokenSweepMode: "fixed-step"
  },
  view: {
    selectedTrendMetric: "prefillTps",
    showFormulaTrace: true,
    showIntermediateMetrics: true,
    showBottleneckBackground: true,
    showTrendDataPoints: true
  }
};

const comparisonColors = ["#2563eb", "#f97316", "#22c55e", "#e11d48"];

function nextComparisonColorIndex(profiles: ComparisonProfile[]) {
  const usedColors = new Set(profiles.map((profile) => profile.color));
  const availableIndex = comparisonColors.findIndex((color) => !usedColors.has(color));
  return availableIndex >= 0 ? availableIndex : profiles.length % comparisonColors.length;
}

function inferPrecision(platform: PlatformInput): PrecisionPresetId {
  const { bytesPerWeight: w, bytesPerExpert: x, bytesPerActivation: a } = platform;
  if (w === 0.5 && x === 0.5 && a === 1) return "w4a8";
  if (w === 1 && x === 1 && a === 1) return "fp8";
  if (w === 2 && x === 2 && a === 2) return "bf16";
  return "custom";
}

function profileLabel(modelId: ModelId, platform: PlatformInput, precision = inferPrecision(platform)) {
  const modelName = getModelDefinition(modelId).displayName.replace(/^.*\//, "");
  return `${modelName} (b${platform.batchSize}, n${platform.chipCount}, ${precision === "custom" ? "custom precision" : precision.toUpperCase()})`;
}

function createProfile(
  modelId: ModelId,
  platform: PlatformInput,
  index: number,
  selectedPrecision?: PrecisionPresetId
): ComparisonProfile {
  const precision = selectedPrecision ?? inferPrecision(platform);
  return {
    id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    label: profileLabel(modelId, platform, precision),
    modelId,
    platform: { ...platform },
    precision,
    color: comparisonColors[index % comparisonColors.length],
    enabled: true
  };
}

function validateState(state: CalculatorState): CalculatorValidation {
  const errors: CalculatorValidation = {};
  const model = getModelDefinition(state.modelId);

  if (state.platform.computeThroughputTflops <= 0) {
    errors.computeThroughputTflops = "需大于 0";
  }

  if (state.platform.memoryBandwidthGbps <= 0) {
    errors.memoryBandwidthGbps = "需大于 0";
  }

  if (state.platform.memoryCapacityGb <= 0) {
    errors.memoryCapacityGb = "需大于 0";
  }

  if (!Number.isInteger(state.platform.chipCount) || state.platform.chipCount < 1) {
    errors.chipCount = "需为大于或等于 1 的整数";
  }

  ([
    "prefillComputeParallelEfficiency",
    "prefillBandwidthParallelEfficiency",
    "decodeComputeParallelEfficiency",
    "decodeBandwidthParallelEfficiency"
  ] as const).forEach((key) => {
    const value = state.platform[key];
    if (!Number.isFinite(value) || value <= 0 || value > 1) errors[key] = "需在 0（不含）到 1 之间";
  });

  if (!Number.isFinite(state.platform.runtimeOverheadGb) || state.platform.runtimeOverheadGb < 0) {
    errors.runtimeOverheadGb = "需为大于或等于 0 的数值";
  }

  if (
    !Number.isFinite(state.platform.prefillCacheTrafficFactor) ||
    state.platform.prefillCacheTrafficFactor < 0 ||
    state.platform.prefillCacheTrafficFactor > 1
  ) {
    errors.prefillCacheTrafficFactor = "需在 0 到 1 之间";
  }

  if (state.workload.prefillTokenLength <= 0) {
    errors.prefillTokenLength = "需大于 0";
  } else if (state.workload.prefillTokenLength > model.contextLimit) {
    errors.prefillTokenLength = `不能超过当前模型最大上下文 ${model.contextLimit.toLocaleString()}`;
  }

  const resolvedDecodeOutputTokens =
    state.workload.decodeOutputTokens ?? state.workload.prefillTokenLength;
  if (!Number.isFinite(resolvedDecodeOutputTokens) || resolvedDecodeOutputTokens < 0) {
    errors.decodeOutputTokens = "需大于或等于 0";
  } else if (state.workload.prefillTokenLength + resolvedDecodeOutputTokens > model.contextLimit) {
    errors.decodeOutputTokens = `Prompt + Decode 不能超过当前模型最大上下文 ${model.contextLimit.toLocaleString()}`;
  }

  if (state.workload.tokenRangeStart > state.workload.tokenRangeEnd) {
    errors.tokenRangeStart = "Start 不能大于 End";
  }

  if (state.workload.tokenRangeStart > model.contextLimit) {
    errors.tokenRangeStart = `不能超过当前模型最大上下文 ${model.contextLimit.toLocaleString()}`;
  }

  if (state.workload.tokenRangeEnd > model.contextLimit) {
    errors.tokenRangeEnd = `不能超过当前模型最大上下文 ${model.contextLimit.toLocaleString()}`;
  }

  if (state.workload.tokenRangeStep <= 0) {
    errors.tokenRangeStep = "Step 需大于 0";
  }

  if (!Number.isFinite(state.workload.logarithmicScaleMultiplier) || state.workload.logarithmicScaleMultiplier <= 1) {
    errors.logarithmicScaleMultiplier = "对数刻度倍率需大于 1";
  }

  const span = state.workload.tokenRangeEnd - state.workload.tokenRangeStart;

  if (state.workload.tokenRangeStep > span && span > 0) {
    errors.tokenRangeStep = "Step 不能大于 End - Start";
  }

  const pointCount = Math.floor(span / state.workload.tokenRangeStep) + 1;

  if (Number.isFinite(pointCount) && pointCount > 500) {
    errors.tokenRangeStep = "趋势点数超过 500，请增大 Step";
  }

  return errors;
}

function resolveDecodeOutputTokens(workload: WorkloadInput): WorkloadInput {
  return {
    ...workload,
    decodeOutputTokens: workload.decodeOutputTokens ?? workload.prefillTokenLength
  };
}

export function useCalculatorState() {
  const [state, setState] = useState<CalculatorState>(defaultState);
  const [calculationRevision, setCalculationRevision] = useState(0);
  const [structureModelId, setStructureModelId] = useState<ModelId>(defaultState.modelId);
  const [formulaModelId, setFormulaModelId] = useState<ModelId>(defaultState.modelId);
  const [calculationSnapshot, setCalculationSnapshot] = useState<CalculationSnapshot>({
    modelId: defaultState.modelId,
    platform: { ...defaultState.platform },
    workload: resolveDecodeOutputTokens(defaultState.workload)
  });
  const [historyRecords, setHistoryRecords] =
    useState<CalculationHistoryRecord[]>(loadHistoryRecords);
  const [result, setResult] = useState<PerformanceResult | null>(() => {
    const model = getModelDefinition(defaultState.modelId);
    return calculatePerformanceResult(
      model,
      defaultState.platform,
      resolveDecodeOutputTokens(defaultState.workload)
    );
  });
  const [status, setStatus] = useState<CalculationStatus>("calculated");
  const [comparisonProfiles, setComparisonProfiles] = useState<ComparisonProfile[]>(() => [
    createProfile(defaultState.modelId, defaultState.platform, 0)
  ]);
  const [comparisonResults, setComparisonResults] = useState<ComparisonResult[]>(() =>
    calculateComparisonResults(
      comparisonProfiles,
      resolveDecodeOutputTokens(defaultState.workload)
    )
  );
  const [comparisonError, setComparisonError] = useState<string | null>(null);

  const validationErrors = useMemo(() => validateState(state), [state]);

  const selectedModel = useMemo(() => getModelDefinition(state.modelId), [state.modelId]);
  const availableFamilies = useMemo(() => getModelFamilies(), []);
  const selectedFamily = selectedModel.family;
  const availableModels = useMemo(
    () => getModelsByFamily(selectedFamily),
    [selectedFamily]
  );
  const structureSelectedModel = useMemo(
    () => getModelDefinition(structureModelId),
    [structureModelId]
  );
  const structureSelectedFamily = structureSelectedModel.family;
  const structureAvailableModels = useMemo(
    () => getModelsByFamily(structureSelectedFamily),
    [structureSelectedFamily]
  );
  const formulaSelectedModel = useMemo(
    () => getModelDefinition(formulaModelId),
    [formulaModelId]
  );
  const formulaSelectedFamily = formulaSelectedModel.family;
  const formulaAvailableModels = useMemo(
    () => getModelsByFamily(formulaSelectedFamily),
    [formulaSelectedFamily]
  );
  const formulaResult = useMemo(() => {
    if (formulaModelId === calculationSnapshot.modelId) {
      return result;
    }

    return calculatePerformanceResult(
      formulaSelectedModel,
      {
        ...calculationSnapshot.platform,
        bytesPerWeight: formulaSelectedModel.recommendedPrecision.bytesPerWeight,
        bytesPerActivation: formulaSelectedModel.recommendedPrecision.bytesPerActivation,
        bytesPerExpert: formulaSelectedModel.recommendedPrecision.bytesPerExpert
      },
      calculationSnapshot.workload
    );
  }, [calculationSnapshot, formulaModelId, formulaSelectedModel, result]);

  function updateModelId(modelId: ModelId) {
    const model = getModelDefinition(modelId);
    setState((current) => {
      const platform = {
        ...current.platform,
        computeThroughputTflops: defaultComputeThroughputTflops(modelId),
        bytesPerWeight: model.recommendedPrecision.bytesPerWeight,
        bytesPerActivation: model.recommendedPrecision.bytesPerActivation,
        bytesPerExpert: model.recommendedPrecision.bytesPerExpert
      };
      return { ...current, modelId, platform, precisionPreset: inferPrecision(platform) };
    });
    setStatus("ready");
  }

  function updateModelFamily(family: string) {
    const [firstModel] = getModelsByFamily(family);

    if (!firstModel) {
      return;
    }

    updateModelId(firstModel.id);
  }

  function updateFormulaModelId(modelId: ModelId) {
    setFormulaModelId(modelId);
  }

  function updateStructureModelId(modelId: ModelId) {
    setStructureModelId(modelId);
  }

  function updateStructureModelFamily(family: string) {
    const [firstModel] = getModelsByFamily(family);

    if (!firstModel) {
      return;
    }

    setStructureModelId(firstModel.id);
  }

  function updateFormulaModelFamily(family: string) {
    const [firstModel] = getModelsByFamily(family);

    if (!firstModel) {
      return;
    }

    setFormulaModelId(firstModel.id);
  }

  function updatePlatform<K extends keyof PlatformInput>(key: K, value: PlatformInput[K]) {
    setState((current) => {
      const platform = { ...current.platform, [key]: value };
      if (key === "chipCount") Object.assign(platform, defaultParallelEfficiencies(Number(value)));
      return {
        ...current,
        precisionPreset: ["bytesPerWeight", "bytesPerExpert", "bytesPerActivation"].includes(key)
          ? "custom"
          : current.precisionPreset,
        platform
      };
    });
    setStatus("ready");
  }

  function applyPrecisionPreset(preset: Exclude<PrecisionPresetId, "custom">) {
    const values = preset === "w4a8"
      ? { bytesPerWeight: 0.5, bytesPerExpert: 0.5, bytesPerActivation: 1 }
      : preset === "bf16"
        ? { bytesPerWeight: 2, bytesPerExpert: 2, bytesPerActivation: 2 }
        : { bytesPerWeight: 1, bytesPerExpert: 1, bytesPerActivation: 1 };
    setState((current) => ({
      ...current,
      precisionPreset: preset,
      platform: { ...current.platform, ...values }
    }));
    setStatus("ready");
  }

  function updateWorkload<K extends keyof WorkloadInput>(key: K, value: WorkloadInput[K]) {
    setState((current) => ({
      ...current,
      workload: {
        ...current.workload,
        [key]: value
      }
    }));
    setStatus("ready");
  }

  function updateView<K extends keyof CalculatorViewState>(
    key: K,
    value: CalculatorViewState[K]
  ) {
    setState((current) => ({
      ...current,
      view: {
        ...current.view,
        [key]: value
      }
    }));
  }

  function applyQuickRange(target: QuickRangeTarget, tokenLength: number) {
    const model = getModelDefinition(state.modelId);

    if (target !== "tokenRangeStep" && tokenLength > model.contextLimit) {
      setStatus("invalid");
      return;
    }

    setState((current) => ({
      ...current,
      workload: {
        ...current.workload,
        [target]: tokenLength
      }
    }));
    setStatus("ready");
  }

  function reset() {
    setState(defaultState);
    const model = getModelDefinition(defaultState.modelId);
    const resolvedWorkload = resolveDecodeOutputTokens(defaultState.workload);
    setResult(
      calculatePerformanceResult(
        model,
        defaultState.platform,
        resolvedWorkload
      )
    );
    setCalculationSnapshot({
      modelId: model.id,
      platform: { ...defaultState.platform },
      workload: resolvedWorkload
    });
    const profile = createProfile(defaultState.modelId, defaultState.platform, 0, defaultState.precisionPreset);
    setComparisonProfiles([profile]);
    setComparisonResults(calculateComparisonResults([profile], resolvedWorkload));
    setComparisonError(null);
    setStatus("calculated");
  }

  function calculate() {
    if (Object.keys(validationErrors).length > 0) {
      setStatus("invalid");
      return;
    }

    setStatus("calculating");
    const model = getModelDefinition(state.modelId);
    const nextResult = calculatePerformanceResult(
      model,
      state.platform,
      resolveDecodeOutputTokens(state.workload)
    );
    setResult(nextResult);
    setStructureModelId(model.id);
    setFormulaModelId(model.id);
    setCalculationSnapshot({
      modelId: model.id,
      platform: { ...state.platform },
      workload: resolveDecodeOutputTokens(state.workload)
    });
    const historyRecord: CalculationHistoryRecord = {
      id:
        globalThis.crypto?.randomUUID?.() ??
        `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      createdAt: new Date().toISOString(),
      modelId: model.id,
      modelFamily: model.family,
      modelDisplayName: model.displayName,
      platform: { ...state.platform },
      workload: resolveDecodeOutputTokens(state.workload),
      result: { ...nextResult.summary }
    };
    setHistoryRecords((current) => {
      const next = [historyRecord, ...current];
      saveHistoryRecords(next);
      return next;
    });
    setCalculationRevision((current) => current + 1);
    setStatus("calculated");
  }

  function addComparisonProfile() {
    setComparisonProfiles((current) => {
      if (current.length >= 4) return current;
      return [...current, createProfile(
        state.modelId,
        state.platform,
        nextComparisonColorIndex(current),
        state.precisionPreset
      )];
    });
  }

  function duplicateComparisonProfile(profileId: string) {
    setComparisonProfiles((current) => {
      if (current.length >= 4) return current;
      const source = current.find((profile) => profile.id === profileId);
      if (!source) return current;
      const copy = createProfile(source.modelId, source.platform, nextComparisonColorIndex(current));
      return [...current, { ...copy, label: `${source.label} copy`, precision: source.precision }];
    });
  }

  function deleteComparisonProfile(profileId: string) {
    setComparisonProfiles((current) => current.filter((profile) => profile.id !== profileId));
  }

  function toggleComparisonProfile(profileId: string) {
    setComparisonProfiles((current) => current.map((profile) =>
      profile.id === profileId ? { ...profile, enabled: !profile.enabled } : profile
    ));
  }

  function updateComparisonProfile(
    profileId: string,
    field: "batchSize" | "chipCount" | "precision",
    value: number | PrecisionPresetId
  ) {
    setComparisonProfiles((current) => current.map((profile) => {
      if (profile.id !== profileId) return profile;
      let platform = { ...profile.platform };
      let precision = profile.precision;
      if (field === "batchSize" || field === "chipCount") {
        platform[field] = Math.max(1, Number(value));
        if (field === "chipCount") Object.assign(platform, defaultParallelEfficiencies(platform.chipCount));
      } else {
        precision = value as PrecisionPresetId;
        const bytes = precision === "custom"
          ? null
          : precision === "w4a8"
          ? { bytesPerWeight: 0.5, bytesPerExpert: 0.5, bytesPerActivation: 1 }
          : precision === "bf16"
            ? { bytesPerWeight: 2, bytesPerExpert: 2, bytesPerActivation: 2 }
            : { bytesPerWeight: 1, bytesPerExpert: 1, bytesPerActivation: 1 };
        if (bytes) platform = { ...platform, ...bytes };
      }
      return { ...profile, platform, precision, label: profileLabel(profile.modelId, platform, precision) };
    }));
  }

  function calculateComparison() {
    try {
      setComparisonResults(calculateComparisonResults(
        comparisonProfiles,
        resolveDecodeOutputTokens(state.workload)
      ));
      setComparisonError(null);
    } catch (error) {
      setComparisonError(error instanceof Error ? error.message : "对比计算失败");
    }
  }

  function clearHistory() {
    setHistoryRecords([]);
    clearStoredHistoryRecords();
  }

  function deleteHistoryRecord(recordId: string) {
    setHistoryRecords((current) => {
      const next = current.filter((record) => record.id !== recordId);
      saveHistoryRecords(next);
      return next;
    });
  }

  return {
    state,
    calculationSnapshot,
    result,
    historyRecords,
    status,
    selectedModel,
    selectedFamily,
    availableFamilies,
    availableModels,
    structureModelId,
    structureSelectedModel,
    structureSelectedFamily,
    structureAvailableModels,
    formulaModelId,
    formulaSelectedModel,
    formulaSelectedFamily,
    formulaAvailableModels,
    formulaResult,
    calculationRevision,
    comparisonProfiles,
    comparisonResults,
    comparisonError,
    validationErrors,
    updateModelFamily,
    updateModelId,
    updateStructureModelFamily,
    updateStructureModelId,
    updateFormulaModelFamily,
    updateFormulaModelId,
    updatePlatform,
    applyPrecisionPreset,
    updateWorkload,
    updateView,
    applyQuickRange,
    reset,
    calculate,
    addComparisonProfile,
    duplicateComparisonProfile,
    deleteComparisonProfile,
    toggleComparisonProfile,
    updateComparisonProfile,
    calculateComparison,
    clearHistory,
    deleteHistoryRecord
  };
}
