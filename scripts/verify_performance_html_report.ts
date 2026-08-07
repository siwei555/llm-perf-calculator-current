import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { getModelDefinition } from "../src/engines/model-registry";
import { calculatePerformanceResult } from "../src/features/performance-calculator/services/performanceCalculator";
import { buildPerformanceHtmlReport } from "../src/features/performance-calculator/services/performanceHtmlReporter";

async function main() {
  const model = getModelDefinition(process.argv[3] ?? "qwen3.6-27b-fp8");
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
      bytesPerWeight: model.recommendedPrecision.bytesPerWeight,
      bytesPerActivation: model.recommendedPrecision.bytesPerActivation,
      bytesPerExpert: model.recommendedPrecision.bytesPerExpert
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
  const html = buildPerformanceHtmlReport({
    model,
    snapshot,
    result,
    exportedAt: new Date("2026-08-05T00:00:00.000Z")
  });

  assert.match(html, /<!doctype html>/i);
  assert.match(html, /charset="utf-8"/i);
  assert.match(html, /Model Config/);
  assert.match(html, /Theoretical compute \/ request/);
  assert.match(html, /Prefill Detail/);
  assert.match(html, /TPS @20% Compute Util/);
  assert.match(html, /TPS @40% Compute Util/);
  assert.match(html, /Decode Detailed Data/);
  assert.match(html, /TPS @40% BW Util/);
  assert.match(html, /TPS @60% BW Util/);
  assert.match(html, /TPS @80% BW Util/);
  assert.doesNotMatch(html, /Util\+MTP/);
  switch (model.formulaStrategyId) {
    case "hybrid-linear-dense":
      assert.match(html, /Dense FFN/);
      assert.match(html, /Gated Attention/);
      assert.match(html, /Gated DeltaNet Attention/);
      assert.doesNotMatch(html, /routed_experts/);
      break;
    case "hybrid-linear-moe":
      assert.match(html, /MoE FFN/);
      assert.match(html, /Gated Attention/);
      assert.match(html, /Gated DeltaNet Attention/);
      assert.match(html, /active_experts \/ token/);
      break;
    case "dense-decoder-transformer":
      assert.match(html, /Dense FFN/);
      assert.match(html, /Sliding-window Attention/);
      assert.match(html, /Full Attention/);
      assert.doesNotMatch(html, /active_experts \/ token/);
      break;
    case "dense-decoder-moe":
      assert.match(html, /MoE FFN/);
      assert.match(html, /Sliding-window Attention/);
      assert.match(html, /Full Attention/);
      assert.match(html, /active_experts \/ token/);
      break;
    case "deepseek-v4-compressed-moe":
      assert.match(html, /Sparse MoE FFN/);
      assert.match(html, /Sliding Attention/);
      assert.match(html, /CSA Attention/);
      assert.match(html, /HCA Attention/);
      assert.match(html, /Indexer projections/);
      break;
  }

  await writeFile(process.argv[2] ?? "performance-report-verification.html", html, "utf8");
}

void main();
