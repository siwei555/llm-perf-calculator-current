import { writeFile } from "node:fs/promises";
import assert from "node:assert/strict";
import { getModelDefinition } from "../src/engines/model-registry";
import { calculatePerformanceResult } from "../src/features/performance-calculator/services/performanceCalculator";
import { buildPerformanceWorkbook } from "../src/features/performance-calculator/services/performanceExcelExporter";

async function main() {
  const model = getModelDefinition("qwen3.6-27b-fp8");
  const snapshot = {
  modelId: model.id,
  platform: {
    computeThroughputTflops: 124,
    memoryBandwidthGbps: 273,
    memoryCapacityGb: 128,
    computeEfficiency: 0.8,
    bandwidthEfficiency: 0.6,
    prefillCacheTrafficFactor: 0.1,
    batchSize: 1,
    runtimeOverheadGb: 4,
    bytesPerWeight: 1,
    bytesPerActivation: 2,
    bytesPerExpert: 1
  },
  workload: {
    prefillTokenLength: 131072,
    decodeOutputTokens: 4096,
    tokenRangeStart: 4096,
    tokenRangeEnd: 131072,
    tokenRangeStep: 4096,
    tokenSweepMode: "fixed-step" as const
  }
  };
  const result = calculatePerformanceResult(model, snapshot.platform, snapshot.workload);
  assert.equal(result.summary.finalDecodeContext, 135168);
  assert.ok(result.summary.initialDecodeTps > result.summary.averageDecodeTps);
  assert.ok(result.summary.decodeTimeMs > 0);
  assert.equal(result.summary.peakRuntimeMemoryGb, result.summary.totalRuntimeMemoryGb);

  const fp8ActivationResult = calculatePerformanceResult(
    model,
    { ...snapshot.platform, bytesPerActivation: 1 },
    snapshot.workload
  );
  const bf16Temp = result.memoryBreakdown.find((row) => row.key === "peakTempWorkingSet")?.valueGb ?? 0;
  const fp8Temp = fp8ActivationResult.memoryBreakdown.find((row) => row.key === "peakTempWorkingSet")?.valueGb ?? 0;
  assert.ok(Math.abs(bf16Temp / fp8Temp - 2) < 1e-9);

  const zeroCacheTrafficResult = calculatePerformanceResult(
    model,
    { ...snapshot.platform, prefillCacheTrafficFactor: 0 },
    snapshot.workload
  );
  const fullCacheTrafficResult = calculatePerformanceResult(
    model,
    { ...snapshot.platform, prefillCacheTrafficFactor: 1 },
    snapshot.workload
  );
  const zeroBandwidthCeiling = Number(zeroCacheTrafficResult.comparisonRows[2]?.prefill);
  const fullBandwidthCeiling = Number(fullCacheTrafficResult.comparisonRows[2]?.prefill);
  assert.ok(zeroBandwidthCeiling > fullBandwidthCeiling);
  const workbook = await buildPerformanceWorkbook({
    model,
    snapshot,
    result,
    exportedAt: new Date("2026-08-03T12:00:00+08:00")
  });
  const buffer = await workbook.xlsx.writeBuffer();
  await writeFile(process.argv[2] ?? "performance-export-verification.xlsx", Buffer.from(buffer));
}

void main();
