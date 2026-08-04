// @ts-nocheck -- standalone Node validation script; the app intentionally does not depend on @types/node
import { writeFile } from "node:fs/promises";
import { getModelDefinition } from "../src/engines/model-registry/index";
import { calculatePerformanceResult } from "../src/features/performance-calculator/services/performanceCalculator";
import { buildPerformanceWorkbook } from "../src/features/performance-calculator/services/performanceExcelExporter";
import type { CalculationSnapshot } from "../src/features/performance-calculator/state/useCalculatorState";

async function main() {
  const outputPath = process.argv[2];

  if (!outputPath) {
    throw new Error("Usage: validate_excel_export.ts <output.xlsx>");
  }

  const model = getModelDefinition("qwen3.6-27b-fp8");
  const snapshot: CalculationSnapshot = {
    modelId: model.id,
    platform: {
      computeThroughputTflops: 124,
      memoryBandwidthGbps: 273,
      memoryCapacityGb: 128,
      computeEfficiency: 0.8,
      bandwidthEfficiency: 0.6,
      batchSize: 1,
      runtimeOverheadGb: 4,
      bytesPerWeight: model.recommendedPrecision.bytesPerWeight,
      bytesPerActivation: model.recommendedPrecision.bytesPerActivation,
      bytesPerExpert: model.recommendedPrecision.bytesPerExpert
    },
    workload: {
      prefillTokenLength: 131072,
      decodeOutputTokens: 8192,
      tokenRangeStart: 4096,
      tokenRangeEnd: 131072,
      tokenRangeStep: 4096,
      tokenSweepMode: "fixed-step"
    }
  };
  const result = calculatePerformanceResult(
    model,
    snapshot.platform,
    snapshot.workload
  );
  const workbook = await buildPerformanceWorkbook({
    model,
    snapshot,
    result,
    exportedAt: new Date("2026-07-31T00:00:00Z")
  });
  const buffer = await workbook.xlsx.writeBuffer();
  await writeFile(outputPath, new Uint8Array(buffer));

  console.log(
    JSON.stringify({
      outputPath,
      worksheets: workbook.worksheets.map((sheet) => sheet.name),
      formulaTraceRows: result.formulaTrace.reduce(
        (total, section) => total + section.rows.length,
        0
      ),
      trendRows: result.tokenSweepSeries.length
    })
  );
}

void main();
