import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { getModelDefinition } from "../src/engines/model-registry";
import { calculatePerformanceResult } from "../src/features/performance-calculator/services/performanceCalculator";
import { buildPerformanceJson } from "../src/features/performance-calculator/services/performanceJsonExporter";

async function main() {
  const model = getModelDefinition("qwen3.6-27b-fp8");
  const snapshot = {
    modelId: model.id,
    platform: {
      computeThroughputTflops: 248,
      memoryBandwidthGbps: 273,
      memoryCapacityGb: 128,
      computeEfficiency: 0.4,
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
  const payload = buildPerformanceJson({
    model,
    snapshot,
    result,
    exportedAt: new Date("2026-08-04T00:00:00.000Z")
  });
  assert.equal(payload.schemaVersion, 1);
  assert.equal(payload.metadata.modelId, model.id);
  assert.equal(payload.platform.computeThroughputTflops, 248);
  assert.equal(payload.platform.computeEfficiency, 0.4);
  assert.equal(payload.prefillProjection.length, 8);
  assert.equal(payload.decodeProjection.length, 8);
  assert.ok(payload.intermediateMetrics.length > 0);
  assert.ok(payload.formulaTrace.length > 0);
  assert.ok(payload.tokenTrend.length > 0);
  assert.equal(typeof payload.prefillProjection[0]?.gflopsPerToken, "number");
  await writeFile(
    process.argv[2] ?? "performance-export-verification.json",
    `${JSON.stringify(payload, null, 2)}\n`,
    "utf8"
  );
}

void main();
