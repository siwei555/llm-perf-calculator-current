import ExcelJS, {
  type Alignment,
  type Border,
  type Cell,
  type Fill,
  type Font,
  type Worksheet
} from "exceljs";
import type { ModelDefinition } from "../../../domain/model/types";
import type { PerformanceResult } from "../../../domain/performance/types";
import type { CalculationSnapshot } from "../state/useCalculatorState";

type PerformanceExcelExportInput = {
  model: ModelDefinition;
  snapshot: CalculationSnapshot;
  result: PerformanceResult;
  exportedAt?: Date;
};

const colors = {
  navy: "17243A",
  blue: "2563EB",
  blueSoft: "EAF1FF",
  slate: "5B6B84",
  border: "D7DFEA",
  panel: "F7F9FC",
  white: "FFFFFF",
  green: "15803D",
  greenSoft: "DCFCE7",
  red: "B91C1C",
  redSoft: "FEE2E2",
  amberSoft: "FEF3C7"
};

const detailGroupColors = [
  "D9EAF7",
  "E8D9FF",
  "DDEBF7",
  "FCE4D6",
  "E2F0D9",
  "FFF2CC",
  "E4DFEC",
  "DDEBF7",
  "F4CCCC"
];

const thinBorder: Partial<Border> = {
  style: "thin",
  color: { argb: colors.border }
};

const titleFont: Partial<Font> = {
  name: "Aptos Display",
  size: 20,
  bold: true,
  color: { argb: colors.navy }
};

const sectionFont: Partial<Font> = {
  name: "Aptos",
  size: 12,
  bold: true,
  color: { argb: colors.white }
};

const headerFont: Partial<Font> = {
  name: "Aptos",
  size: 10,
  bold: true,
  color: { argb: colors.navy }
};

const bodyFont: Partial<Font> = {
  name: "Aptos",
  size: 10,
  color: { argb: colors.navy }
};

const sectionFill: Fill = {
  type: "pattern",
  pattern: "solid",
  fgColor: { argb: colors.blue }
};

const headerFill: Fill = {
  type: "pattern",
  pattern: "solid",
  fgColor: { argb: colors.blueSoft }
};

const center: Partial<Alignment> = {
  horizontal: "center",
  vertical: "middle"
};

function applyBaseSheetStyle(sheet: Worksheet) {
  sheet.views = [{ state: "frozen", ySplit: 2, showGridLines: false }];
  sheet.properties.defaultRowHeight = 18;
}

function forEachCell(
  sheet: Worksheet,
  startRow: number,
  startColumn: number,
  endRow: number,
  endColumn: number,
  callback: (cell: Cell) => void
) {
  for (let row = startRow; row <= endRow; row += 1) {
    for (let column = startColumn; column <= endColumn; column += 1) {
      callback(sheet.getCell(row, column));
    }
  }
}

function setBottomBorder(
  sheet: Worksheet,
  row: number,
  startColumn: number,
  endColumn: number
) {
  forEachCell(sheet, row, startColumn, row, endColumn, (cell) => {
    cell.border = { bottom: thinBorder };
  });
}

function styleTitle(sheet: Worksheet, range: string, title: string, subtitle: string) {
  sheet.mergeCells(range);
  const titleCell = sheet.getCell(range.split(":")[0]);
  titleCell.value = title;
  titleCell.font = titleFont;
  titleCell.alignment = { vertical: "middle" };
  titleCell.fill = {
    type: "pattern",
    pattern: "solid",
    fgColor: { argb: colors.panel }
  };
  titleCell.border = {
    bottom: { style: "medium", color: { argb: colors.blue } }
  };
  titleCell.note = subtitle;
  sheet.getRow(Number(titleCell.row)).height = 34;
}

function styleSection(sheet: Worksheet, row: number, startColumn: number, endColumn: number) {
  forEachCell(sheet, row, startColumn, row, endColumn, (cell) => {
    cell.fill = sectionFill;
    cell.font = sectionFont;
    cell.alignment = { vertical: "middle" };
    cell.border = {
      top: thinBorder,
      bottom: thinBorder
    };
  });
  sheet.getRow(row).height = 23;
}

function styleHeaderRow(sheet: Worksheet, row: number, startColumn: number, endColumn: number) {
  forEachCell(sheet, row, startColumn, row, endColumn, (cell) => {
    cell.fill = headerFill;
    cell.font = headerFont;
    cell.alignment = center;
    cell.border = {
      top: thinBorder,
      bottom: thinBorder
    };
  });
  sheet.getRow(row).height = 22;
}

function addLabelValue(
  sheet: Worksheet,
  row: number,
  label: string,
  value: string | number | boolean | Date,
  unit = "",
  source = ""
) {
  sheet.getCell(row, 1).value = label;
  sheet.getCell(row, 1).font = headerFont;
  sheet.getCell(row, 2).value = value;
  sheet.getCell(row, 3).value = source || null;
  applyValueUnitFormat(sheet.getCell(row, 2), unit);
  setBottomBorder(sheet, row, 1, 3);
}

function applyValueUnitFormat(cell: Cell, unit: string) {
  const formatByUnit: Record<string, string> = {
    ms: '#,##0.00 "ms"',
    "tokens/s": '#,##0.00 "tokens/s"',
    GB: '#,##0.000 "GB"',
    layers: '#,##0 "layers"',
    heads: '#,##0 "heads"',
    experts: '#,##0 "experts"',
    tokens: '#,##0 "tokens"',
    TFLOPS: '#,##0.00 "TFLOPS"',
    "GB/s": '#,##0.00 "GB/s"',
    sequences: '#,##0 "sequences"',
    "bytes/param": '#,##0.00 "bytes/param"',
    "bytes/element": '#,##0.00 "bytes/element"'
  };

  if (unit === "ratio") {
    cell.numFmt = "0.0%";
  } else if (formatByUnit[unit]) {
    cell.numFmt = formatByUnit[unit];
  }
}

function setHyperlink(cell: Cell, text: string, hyperlink?: string) {
  if (!hyperlink) {
    cell.value = text || "—";
    return;
  }
  cell.value = { text, hyperlink };
  cell.font = {
    ...bodyFont,
    color: { argb: colors.blue },
    underline: true
  };
}

function numericValue(value: string) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : value;
}

function safeFilePart(value: string) {
  return value
    .replace(/[<>:"/\\|?*\u0000-\u001F]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 80);
}

type DetailRow = {
  item: string;
  value: string | number | boolean | Date;
  unit?: string;
  notes: string;
};

type DetailGroup = {
  type: string;
  rows: DetailRow[];
};

function addEstimateDetailSheet(
  workbook: ExcelJS.Workbook,
  model: ModelDefinition,
  snapshot: CalculationSnapshot,
  result: PerformanceResult,
  exportedAt: Date
) {
  const sheet = workbook.addWorksheet("Estimate Detail");
  sheet.columns = [
    { width: 25 },
    { width: 36 },
    { width: 28 },
    { width: 86 }
  ];

  styleTitle(
    sheet,
    "A1:D1",
    `${model.displayName} Performance Estimate Detail`,
    "Grouped calculation inputs, results, intermediate values, and formula trace."
  );
  sheet.mergeCells("A2:D2");
  sheet.getCell("A2").value =
    "工程估算明细：Value 将数值与单位放在同一单元格；Notes 给出字段说明、计算来源或实际使用的公式。";
  sheet.getCell("A2").font = { ...bodyFont, italic: true, color: { argb: colors.slate } };
  sheet.getCell("A2").alignment = { wrapText: true, vertical: "middle" };
  sheet.getRow(2).height = 30;
  sheet.getRow(3).values = ["Type", "Item", "Value", "Notes (Description / Formula)"];
  styleHeaderRow(sheet, 3, 1, 4);

  const { platform, workload } = snapshot;
  const groups: DetailGroup[] = [
    {
      type: "Model Parameters",
      rows: [
        { item: "model", value: model.displayName, notes: "Selected model display name" },
        { item: "model_family", value: model.family, notes: "Model registry family" },
        { item: "formula_strategy", value: model.formulaStrategyId, notes: "Formula strategy selected by model registry" },
        { item: "hidden_size (H)", value: model.hiddenSize, notes: "Hidden dimension from model config" },
        { item: "num_layers", value: model.decoderLayers, unit: "layers", notes: "Decoder layer count from model config" },
        { item: "attention_heads", value: model.attentionHeads, unit: "heads", notes: "Query attention heads" },
        { item: "kv_heads", value: model.kvHeads, unit: "heads", notes: "Key/value attention heads" },
        { item: "head_dim", value: model.headDim, notes: "Attention head dimension" },
        { item: "routed_experts", value: model.moeExperts, unit: "experts", notes: "Routed experts per MoE layer" },
        { item: "active_experts_per_token", value: model.activeExperts, unit: "experts", notes: "Experts activated for each token" },
        { item: "moe_intermediate_size", value: model.moeIntermediateSize, notes: "MoE expert intermediate dimension" },
        { item: "context_limit", value: model.contextLimit, unit: "tokens", notes: "Maximum supported context length" }
      ]
    },
    {
      type: "Workload",
      rows: [
        { item: "prompt_length", value: workload.prefillTokenLength, unit: "tokens", notes: "Prefill sequence length (S)" },
        { item: "decode_output_tokens", value: workload.decodeOutputTokens ?? workload.prefillTokenLength, unit: "tokens", notes: "Decode output length; falls back to prompt length when empty" },
        { item: "batch_size", value: platform.batchSize, unit: "sequences", notes: "Concurrent sequences used by the estimate" },
        { item: "token_sweep", value: `${workload.tokenRangeStart.toLocaleString()} – ${workload.tokenRangeEnd.toLocaleString()} / ${workload.tokenRangeStep.toLocaleString()} tokens`, notes: "Start – end / step for the Token Trend sheet" }
      ]
    },
    {
      type: "Platform Assumptions",
      rows: [
        { item: "compute_throughput", value: platform.computeThroughputTflops, unit: "TFLOPS", notes: "Peak platform compute input" },
        { item: "compute_efficiency", value: platform.computeEfficiency, unit: "ratio", notes: "Calibratable efficiency assumption" },
        { item: "effective_compute", value: platform.computeThroughputTflops * platform.computeEfficiency, unit: "TFLOPS", notes: "compute_throughput × compute_efficiency" },
        { item: "memory_bandwidth", value: platform.memoryBandwidthGbps, unit: "GB/s", notes: "Peak platform memory bandwidth input" },
        { item: "bandwidth_efficiency", value: platform.bandwidthEfficiency, unit: "ratio", notes: "Calibratable bandwidth efficiency assumption" },
        { item: "effective_bandwidth", value: platform.memoryBandwidthGbps * platform.bandwidthEfficiency, unit: "GB/s", notes: "memory_bandwidth × bandwidth_efficiency" },
        { item: "prefill_cache_traffic_factor", value: platform.prefillCacheTrafficFactor, unit: "ratio", notes: "Editable factor in B_prefill = B_weights + M_cache × factor; default 0.10, allowed range 0–1" },
        { item: "memory_capacity", value: platform.memoryCapacityGb, unit: "GB", notes: "Available HBM / VRAM capacity" },
        { item: "runtime_overhead", value: platform.runtimeOverheadGb, unit: "GB", notes: "Framework and runtime reservation" },
        { item: "bytes_per_weight", value: platform.bytesPerWeight, unit: "bytes/param", notes: "Weight precision assumption" },
        { item: "bytes_per_activation", value: platform.bytesPerActivation, unit: "bytes/element", notes: "Activation and cache precision assumption" },
        { item: "bytes_per_expert", value: platform.bytesPerExpert, unit: "bytes/param", notes: "Expert weight precision assumption" }
      ]
    },
    {
      type: "Core Results",
      rows: [
        { item: "TTFT", value: result.summary.ttftMs, unit: "ms", notes: "Time to first token from the calculated prefill path" },
        { item: "Prefill TPS", value: result.summary.prefillTps, unit: "tokens/s", notes: "min(prefill compute ceiling, prefill bandwidth ceiling)" },
        { item: "Initial Decode TPS", value: result.summary.initialDecodeTps, unit: "tokens/s", notes: "Decode TPS at the prompt-length context" },
        { item: "Average Decode TPS", value: result.summary.averageDecodeTps, unit: "tokens/s", notes: "N_decode / sum_t(1 / TPS_decode(S_prompt + t))" },
        { item: "Total Decode Time", value: result.summary.decodeTimeMs, unit: "ms", notes: "Accumulated latency across all generated tokens" },
        { item: "Final Decode Context", value: result.summary.finalDecodeContext, unit: "tokens", notes: "Prompt length + decode output tokens" },
        { item: "Peak Runtime Memory", value: result.summary.peakRuntimeMemoryGb, unit: "GB", notes: "Memory at the final decode context" },
        { item: "Prefill Bottleneck", value: result.summary.prefillBottleneck, notes: "Binding ceiling for prefill" },
        { item: "Decode Bottleneck", value: result.summary.decodeBottleneck, notes: "Binding ceiling for decode" },
        { item: "Fits Capacity", value: result.summary.memoryFitsCapacity ? "Yes" : "No", notes: "Runtime Memory ≤ memory_capacity" }
      ]
    },
    {
      type: "Memory Breakdown",
      rows: result.memoryBreakdown.map((item) => ({
        item: item.label,
        value: item.valueGb,
        unit: "GB",
        notes:
          item.key === "estimatedTotal"
            ? "Sum of weights, persistent decode cache/state, peak temporary working set, and runtime overhead"
            : item.key === "runtimeOverhead"
              ? "Editable runtime/framework assumption"
              : "Calculated memory component"
      }))
    },
    {
      type: "Intermediate Metrics",
      rows: result.intermediateMetrics.map((item) => ({
        item: `${item.label} (${item.symbol})`,
        value: `${item.value}${item.unit ? ` ${item.unit}` : ""}`,
        notes: `Source: ${item.source}`
      }))
    },
    ...result.formulaTrace.map<DetailGroup>((section) => ({
      type: `${section.category.toUpperCase()} Formula Trace`,
      rows: section.rows.map((item) => ({
        item: item.label,
        value: item.evaluated,
        notes: [item.expression, item.sourceLabel, item.sourceUrl].filter(Boolean).join(" | ")
      }))
    })),
    {
      type: "Export Metadata",
      rows: [
        { item: "exported_at", value: exportedAt, notes: "Local export time" },
        { item: "model_parameter_source", value: model.parameterSourceUrl ?? "—", notes: "Model configuration source URL" },
        { item: "weight_source", value: model.weightSourceUrl ?? "—", notes: "Model weight source URL" }
      ]
    }
  ];

  let row = 4;
  groups.forEach((group, groupIndex) => {
    const startRow = row;
    group.rows.forEach((item) => {
      sheet.getCell(row, 2).value = item.item;
      sheet.getCell(row, 3).value = item.value;
      sheet.getCell(row, 4).value = item.notes;
      applyValueUnitFormat(sheet.getCell(row, 3), item.unit ?? "");
      if (item.value instanceof Date) {
        sheet.getCell(row, 3).numFmt = "yyyy-mm-dd hh:mm:ss";
      }
      row += 1;
    });
    const endRow = row - 1;
    sheet.getCell(startRow, 1).value = group.type;
    sheet.getCell(startRow, 1).font = { ...headerFont, bold: true };
    sheet.getCell(startRow, 1).alignment = { horizontal: "center", vertical: "middle", wrapText: true };

    const fillColor = detailGroupColors[groupIndex % detailGroupColors.length];
    forEachCell(sheet, startRow, 2, endRow, 4, (cell) => {
      cell.border = {
        top: thinBorder,
        bottom: thinBorder,
        left: thinBorder,
        right: thinBorder
      };
      cell.alignment = {
        ...cell.alignment,
        vertical: "middle",
        wrapText: true
      };
    });
    forEachCell(sheet, startRow, 2, endRow, 3, (cell) => {
      cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: fillColor } };
    });
    sheet.getCell(startRow, 1).fill = {
      type: "pattern",
      pattern: "solid",
      fgColor: { argb: fillColor }
    };
    sheet.getCell(startRow, 1).border = {
      top: thinBorder,
      bottom: thinBorder,
      left: thinBorder,
      right: thinBorder
    };
    if (endRow > startRow) {
      sheet.mergeCells(startRow, 1, endRow, 1);
    }
  });

  sheet.autoFilter = { from: "A3", to: `D${row - 1}` };
  sheet.views = [{ state: "frozen", ySplit: 3, showGridLines: false }];
  sheet.pageSetup = {
    orientation: "landscape",
    fitToPage: true,
    fitToWidth: 1,
    fitToHeight: 0,
    margins: { left: 0.25, right: 0.25, top: 0.5, bottom: 0.5, header: 0.2, footer: 0.2 }
  };
  return sheet;
}

function addOverviewSheet(
  workbook: ExcelJS.Workbook,
  model: ModelDefinition,
  snapshot: CalculationSnapshot,
  result: PerformanceResult,
  exportedAt: Date
) {
  const sheet = workbook.addWorksheet("Overview");
  sheet.columns = [
    { key: "label", width: 30 },
    { key: "value", width: 34 },
    { key: "source", width: 62 }
  ];

  styleTitle(
    sheet,
    "A1:C1",
    "LLM Performance Estimate",
    "Snapshot exported from LLM Perf Calculator. Recalculate in the app after changing assumptions."
  );
  sheet.mergeCells("A2:C2");
  sheet.getCell("A2").value =
    "工程估算快照：结果由模型结构、平台有效算力/带宽、Prompt/Decode 长度与精度假设共同决定。";
  sheet.getCell("A2").font = { ...bodyFont, italic: true, color: { argb: colors.slate } };
  sheet.getCell("A2").alignment = { wrapText: true, vertical: "middle" };
  sheet.getRow(2).height = 32;

  sheet.mergeCells("A4:C4");
  sheet.getCell("A4").value = "Core Results";
  styleSection(sheet, 4, 1, 3);

  const kpis = [
    ["TTFT", result.summary.ttftMs, "ms"],
    ["Prefill TPS", result.summary.prefillTps, "tokens/s"],
    ["Average Decode TPS", result.summary.averageDecodeTps, "tokens/s"],
    ["Peak Runtime Memory", result.summary.peakRuntimeMemoryGb, "GB"],
    ["Prefill Bottleneck", result.summary.prefillBottleneck, ""],
    ["Decode Bottleneck", result.summary.decodeBottleneck, ""],
    ["Fits Capacity", result.summary.memoryFitsCapacity ? "Yes" : "No", ""]
  ] as const;

  kpis.forEach(([label, value, unit], index) => {
    const row = 5 + index;
    addLabelValue(sheet, row, label, value, unit, "Calculated result");
  });
  sheet.getCell("B5").value = {
    formula: "'Results'!B9",
    result: result.summary.ttftMs
  };
  sheet.getCell("B6").value = {
    formula: "'Results'!B8",
    result: result.summary.prefillTps
  };
  sheet.getCell("B7").value = {
    formula: "'Results'!C8",
    result: result.summary.decodeTps
  };
  sheet.getCell("B8").value = {
    formula: "'Results'!B18",
    result: result.summary.totalRuntimeMemoryGb
  };
  sheet.getCell("B11").value = {
    formula: 'IF(B8<=B35,"Yes","No")',
    result: result.summary.memoryFitsCapacity ? "Yes" : "No"
  };
  sheet.getCell("B11").fill = result.summary.memoryFitsCapacity
    ? { type: "pattern", pattern: "solid", fgColor: { argb: colors.greenSoft } }
    : { type: "pattern", pattern: "solid", fgColor: { argb: colors.redSoft } };
  sheet.getCell("B11").font = {
    ...headerFont,
    color: { argb: result.summary.memoryFitsCapacity ? colors.green : colors.red }
  };

  sheet.mergeCells("A13:C13");
  sheet.getCell("A13").value = "Model & Calculation Snapshot";
  styleSection(sheet, 13, 1, 3);

  addLabelValue(sheet, 14, "Exported At", exportedAt, "", "Local browser time");
  sheet.getCell("B14").numFmt = "yyyy-mm-dd hh:mm:ss";
  addLabelValue(sheet, 15, "Model", model.displayName, "", "Model registry");
  addLabelValue(sheet, 16, "Model Family", model.family, "", "Model registry");
  addLabelValue(sheet, 17, "Formula Strategy", model.formulaStrategyId, "", "Formula strategy registry");
  addLabelValue(sheet, 18, "Decoder Layers", model.decoderLayers, "layers", "Model config.json");
  addLabelValue(sheet, 19, "Hidden Size", model.hiddenSize, "", "Model config.json");
  addLabelValue(sheet, 20, "Attention Heads", model.attentionHeads, "heads", "Model config.json");
  addLabelValue(sheet, 21, "KV Heads", model.kvHeads, "heads", "Model config.json");
  addLabelValue(sheet, 22, "Routed Experts / Layer", model.moeExperts, "experts", "Model config.json");
  addLabelValue(sheet, 23, "Active Experts / Token", model.activeExperts, "experts", "Model config.json");
  addLabelValue(sheet, 24, "Context Limit", model.contextLimit, "tokens", "Model config.json");
  setHyperlink(sheet.getCell("B25"), "Model parameter source", model.parameterSourceUrl);
  sheet.getCell("A25").value = "Parameter Source";
  sheet.getCell("A25").font = headerFont;
  setBottomBorder(sheet, 25, 1, 3);
  setHyperlink(sheet.getCell("B26"), "Weight file source", model.weightSourceUrl);
  sheet.getCell("A26").value = "Weight Source";
  sheet.getCell("A26").font = headerFont;
  setBottomBorder(sheet, 26, 1, 3);

  sheet.mergeCells("A28:C28");
  sheet.getCell("A28").value = "Platform & Workload Assumptions";
  styleSection(sheet, 28, 1, 3);

  const { platform, workload } = snapshot;
  const assumptions: Array<[string, number | string, string, string]> = [
    ["Compute Throughput", platform.computeThroughputTflops, "TFLOPS", "User platform input"],
    ["Compute Efficiency", platform.computeEfficiency, "ratio", "Engineering assumption"],
    ["Effective Compute", platform.computeThroughputTflops * platform.computeEfficiency, "TFLOPS", "Compute Throughput × Compute Efficiency"],
    ["Memory Bandwidth", platform.memoryBandwidthGbps, "GB/s", "User platform input"],
    ["Bandwidth Efficiency", platform.bandwidthEfficiency, "ratio", "Engineering assumption"],
    ["Effective Bandwidth", platform.memoryBandwidthGbps * platform.bandwidthEfficiency, "GB/s", "Memory Bandwidth × Bandwidth Efficiency"],
    ["Memory Capacity", platform.memoryCapacityGb, "GB", "User platform input"],
    ["Runtime Overhead", platform.runtimeOverheadGb, "GB", "Editable engineering assumption"],
    ["Batch Size", platform.batchSize, "sequences", "User workload input"],
    ["Bytes / Weight", platform.bytesPerWeight, "bytes/param", "Precision assumption"],
    ["Bytes / Activation", platform.bytesPerActivation, "bytes/element", "Precision assumption"],
    ["Bytes / Expert", platform.bytesPerExpert, "bytes/param", "Precision assumption"],
    ["Prompt Length", workload.prefillTokenLength, "tokens", "Calculation workload"],
    ["Decode Output Tokens", workload.decodeOutputTokens ?? workload.prefillTokenLength, "tokens", "Calculation workload"],
    ["Token Sweep Start", workload.tokenRangeStart, "tokens", "Trend workload"],
    ["Token Sweep End", workload.tokenRangeEnd, "tokens", "Trend workload"],
    ["Token Sweep Step", workload.tokenRangeStep, "tokens", "Trend workload"]
    ,["Prefill Cache Traffic Factor", platform.prefillCacheTrafficFactor, "ratio", "Default 0.10; allowed range 0–1"]
  ];

  assumptions.forEach(([label, value, unit, source], index) => {
    const row = 29 + index;
    addLabelValue(sheet, row, label, value, unit, source);
    if (typeof value === "number" && !unit) {
      sheet.getCell(row, 2).numFmt = "#,##0.00";
    }
  });
  sheet.getCell("B31").value = {
    formula: "B29*B30",
    result: platform.computeThroughputTflops * platform.computeEfficiency
  };
  sheet.getCell("B34").value = {
    formula: "B32*B33",
    result: platform.memoryBandwidthGbps * platform.bandwidthEfficiency
  };

  forEachCell(sheet, 1, 1, 46, 3, (cell) => {
    cell.alignment = { ...cell.alignment, vertical: "top", wrapText: true };
  });
  forEachCell(sheet, 5, 3, 46, 3, (cell) => {
    cell.alignment = { ...cell.alignment, horizontal: "left", indent: 1 };
  });
  applyBaseSheetStyle(sheet);
  return sheet;
}

function addResultsSheet(workbook: ExcelJS.Workbook, result: PerformanceResult) {
  const sheet = workbook.addWorksheet("Results");
  sheet.columns = [
    { width: 34 },
    { width: 20 },
    { width: 20 },
    { width: 18 },
    { width: 4 },
    { width: 30 },
    { width: 18 },
    { width: 18 },
    { width: 18 }
  ];
  styleTitle(sheet, "A1:I1", "Results & Intermediate Metrics", "Calculated snapshot values.");

  sheet.mergeCells("A3:D3");
  sheet.getCell("A3").value = "Performance Comparison";
  styleSection(sheet, 3, 1, 4);
  sheet.getRow(4).values = ["Metric", "Prefill", "Decode", "Unit"];
  styleHeaderRow(sheet, 4, 1, 4);
  result.comparisonRows.forEach((item, index) => {
    const row = 5 + index;
    sheet.getRow(row).values = [
      item.label,
      numericValue(item.prefill),
      numericValue(item.decode),
      item.unit
    ];
    setBottomBorder(sheet, row, 1, 4);
    sheet.getCell(row, 2).numFmt = "#,##0.00";
    sheet.getCell(row, 3).numFmt = "#,##0.00";
  });
  sheet.getCell("B8").value = {
    formula: "MIN(B6,B7)",
    result: result.summary.prefillTps
  };
  sheet.getCell("C8").value = {
    formula: "MIN(C6,C7)",
    result: result.summary.decodeTps
  };

  const memoryStart = 12;
  sheet.mergeCells(`A${memoryStart}:D${memoryStart}`);
  sheet.getCell(`A${memoryStart}`).value = "Runtime Memory Breakdown";
  styleSection(sheet, memoryStart, 1, 4);
  sheet.getRow(memoryStart + 1).values = ["Component", "Value", "Unit", "Estimation Role"];
  styleHeaderRow(sheet, memoryStart + 1, 1, 4);
  result.memoryBreakdown.forEach((item, index) => {
    const row = memoryStart + 2 + index;
    const role =
      item.key === "estimatedTotal"
        ? "Weights + persistent cache/state + temporary peak + runtime overhead"
        : item.key === "runtimeOverhead"
          ? "Editable runtime/framework assumption"
          : "Calculated component";
    sheet.getRow(row).values = [item.label, item.valueGb, "GB", role];
    sheet.getCell(row, 2).numFmt = "#,##0.000";
    setBottomBorder(sheet, row, 1, 4);
  });
  sheet.getCell("B18").value = {
    formula: "SUM(B14:B17)",
    result: result.summary.totalRuntimeMemoryGb
  };

  sheet.mergeCells("F3:I3");
  sheet.getCell("F3").value = "Intermediate Metrics";
  styleSection(sheet, 3, 6, 9);
  sheet.getRow(4).getCell(6).value = "Metric";
  sheet.getRow(4).getCell(7).value = "Symbol";
  sheet.getRow(4).getCell(8).value = "Value";
  sheet.getRow(4).getCell(9).value = "Source";
  styleHeaderRow(sheet, 4, 6, 9);
  result.intermediateMetrics.forEach((item, index) => {
    const row = 5 + index;
    sheet.getCell(row, 6).value = item.label;
    sheet.getCell(row, 7).value = item.symbol;
    sheet.getCell(row, 8).value = `${item.value} ${item.unit}`.trim();
    sheet.getCell(row, 9).value = item.source;
    setBottomBorder(sheet, row, 6, 9);
  });

  const lastRow = Math.max(
    memoryStart + result.memoryBreakdown.length + 2,
    result.intermediateMetrics.length + 5
  );
  forEachCell(sheet, 1, 1, lastRow, 9, (cell) => {
    cell.alignment = { ...cell.alignment, vertical: "top", wrapText: true };
  });
  applyBaseSheetStyle(sheet);
  return sheet;
}

function addFormulaTraceSheet(workbook: ExcelJS.Workbook, result: PerformanceResult) {
  const sheet = workbook.addWorksheet("Formula Trace");
  sheet.columns = [
    { width: 14 },
    { width: 42 },
    { width: 64 },
    { width: 45 },
    { width: 22 },
    { width: 58 }
  ];
  styleTitle(
    sheet,
    "A1:F1",
    "Formula Trace",
    "Expressions and evaluated values are exported exactly from the calculator result trace."
  );
  sheet.mergeCells("A2:F2");
  sheet.getCell("A2").value =
    "估算逻辑：Raw ceilings 由有效算力/带宽与每 token 工作量决定；最终吞吐取相关上限的最小值。Memory 为权重、持久缓存/状态、临时峰值和运行时开销之和。";
  sheet.getCell("A2").alignment = { wrapText: true, vertical: "middle" };
  sheet.getCell("A2").font = { ...bodyFont, italic: true, color: { argb: colors.slate } };
  sheet.getRow(2).height = 36;

  sheet.getRow(4).values = [
    "Stage",
    "Formula Step",
    "Expression",
    "Evaluated Result",
    "Source Label",
    "Source URL"
  ];
  styleHeaderRow(sheet, 4, 1, 6);

  let row = 5;
  result.formulaTrace.forEach((section) => {
    section.rows.forEach((item) => {
      sheet.getRow(row).values = [
        section.category.toUpperCase(),
        item.label,
        item.expression,
        item.evaluated,
        item.sourceLabel || null,
        item.sourceUrl || null
      ];
      sheet.getCell(row, 1).font = { ...headerFont, color: { argb: colors.blue } };
      sheet.getCell(row, 3).font = { name: "Cascadia Mono", size: 9, color: { argb: colors.blue } };
      setBottomBorder(sheet, row, 1, 6);
      forEachCell(sheet, row, 1, row, 6, (cell) => {
        cell.alignment = { vertical: "top", wrapText: true };
      });
      row += 1;
    });
  });

  sheet.autoFilter = { from: "A4", to: `F${row - 1}` };
  applyBaseSheetStyle(sheet);
  return sheet;
}

function selectProjectionPoints(result: PerformanceResult, limit = 8) {
  const points = result.projectionSeries;
  if (points.length <= limit) {
    return points;
  }
  return Array.from({ length: limit }, (_, index) =>
    points[Math.round((index * (points.length - 1)) / (limit - 1))]
  );
}

function styleProjectionHeader(sheet: Worksheet, row: number, endColumn: number) {
  forEachCell(sheet, row, 1, row, endColumn, (cell) => {
    cell.fill = { type: "pattern", pattern: "solid", fgColor: { argb: "1F4EAA" } };
    cell.font = { ...headerFont, color: { argb: colors.white } };
    cell.alignment = { horizontal: "center", vertical: "middle", wrapText: true };
    cell.border = { top: thinBorder, bottom: thinBorder, left: thinBorder, right: thinBorder };
  });
  sheet.getRow(row).height = 30;
}

function addPrefillProjectionSheet(
  workbook: ExcelJS.Workbook,
  model: ModelDefinition,
  result: PerformanceResult
) {
  const sheet = workbook.addWorksheet("Prefill Projection");
  sheet.columns = [
    { width: 18 },
    { width: 24 },
    { width: 18 },
    { width: 18 },
    { width: 20 }
  ];
  const points = selectProjectionPoints(result);
  sheet.mergeCells("A1:E1");
  sheet.getCell("A1").value = `${model.displayName} Prefill Detail`;
  styleSection(sheet, 1, 1, 5);
  const detailHeaderRow = 2;
  sheet.getRow(detailHeaderRow).values = [
    "Context",
    "GFLOPs / Token",
    "TPS @20%",
    "TPS @40%",
    "TTFT (sec) @40%"
  ];
  styleProjectionHeader(sheet, detailHeaderRow, 5);
  points.forEach((point, index) => {
    const row = detailHeaderRow + 1 + index;
    sheet.getRow(row).values = [
      point.contextLength,
      point.prefillGflopsPerToken,
      point.prefillTps20,
      point.prefillTps40,
      point.prefillTtftSec40
    ];
    sheet.getCell(row, 1).numFmt = "#,##0";
    forEachCell(sheet, row, 2, row, 5, (cell) => {
      cell.numFmt = "#,##0.000";
    });
    setBottomBorder(sheet, row, 1, 5);
  });
  sheet.views = [{ state: "frozen", ySplit: detailHeaderRow, showGridLines: false }];
  sheet.autoFilter = { from: "A2", to: `E${2 + points.length}` };
  sheet.pageSetup = { orientation: "landscape", fitToPage: true, fitToWidth: 1, fitToHeight: 1 };
  return sheet;
}

function addDecodeProjectionSheet(
  workbook: ExcelJS.Workbook,
  model: ModelDefinition,
  snapshot: CalculationSnapshot,
  result: PerformanceResult
) {
  const sheet = workbook.addWorksheet("Decode Projection");
  sheet.columns = [
    { width: 16 },
    { width: 22 },
    { width: 20 },
    { width: 20 },
    { width: 18 },
    { width: 18 },
    { width: 18 }
  ];
  styleTitle(
    sheet,
    "A1:G1",
    `${model.displayName} Decode Detailed Data`,
    "Single-token decode projections across context lengths; no unmodeled MTP multiplier is applied."
  );
  sheet.mergeCells("A2:G2");
  sheet.getCell("A2").value =
    `TPS@40% / @60% / @80% simultaneously apply the displayed efficiency to compute and bandwidth ceilings. Memory uses Batch ${snapshot.platform.batchSize} and the selected precision assumptions.`;
  sheet.getCell("A2").alignment = { wrapText: true, vertical: "middle" };
  sheet.getCell("A2").font = { ...bodyFont, italic: true, color: { argb: colors.slate } };
  sheet.getRow(2).height = 32;

  const points = selectProjectionPoints(result);
  sheet.getRow(4).values = [
    "Context",
    "Persistent Cache (GB)",
    "Temp Peak (GB)",
    "Total Memory (GB)",
    "TPS @40%",
    "TPS @60%",
    "TPS @80%"
  ];
  styleProjectionHeader(sheet, 4, 7);
  points.forEach((point, index) => {
    const row = 5 + index;
    sheet.getRow(row).values = [
      point.contextLength,
      point.persistentCacheGb,
      point.temporaryMemoryGb,
      point.totalMemoryGb,
      point.decodeTps40,
      point.decodeTps60,
      point.decodeTps80
    ];
    sheet.getCell(row, 1).numFmt = "#,##0";
    forEachCell(sheet, row, 2, row, 7, (cell) => {
      cell.numFmt = "#,##0.000";
    });
    setBottomBorder(sheet, row, 1, 7);
  });
  sheet.autoFilter = { from: "A4", to: `G${4 + points.length}` };
  sheet.views = [{ state: "frozen", ySplit: 4, showGridLines: false }];
  return sheet;
}

function addTrendSheet(workbook: ExcelJS.Workbook, result: PerformanceResult) {
  const sheet = workbook.addWorksheet("Token Trend");
  sheet.columns = [
    { width: 18 },
    { width: 18 },
    { width: 18 },
    { width: 18 },
    { width: 24 },
    { width: 22 },
    { width: 22 }
  ];
  styleTitle(
    sheet,
    "A1:G1",
    "Token Length Trend",
    "All trend points from the current fixed-step sweep."
  );
  sheet.getRow(3).values = [
    "Token Length",
    "Prefill TPS",
    "Initial Decode TPS",
    "TTFT (ms)",
    "Runtime Memory (GB)",
    "Prefill Bottleneck",
    "Decode Bottleneck"
  ];
  styleHeaderRow(sheet, 3, 1, 7);
  result.tokenSweepSeries.forEach((point, index) => {
    const row = 4 + index;
    sheet.getRow(row).values = [
      point.tokenLength,
      point.prefillTps,
      point.decodeTps,
      point.ttftMs,
      point.totalRuntimeMemoryGb,
      point.prefillBottleneck,
      point.decodeBottleneck
    ];
    sheet.getCell(row, 1).numFmt = "#,##0";
    forEachCell(sheet, row, 2, row, 5, (cell) => {
      cell.numFmt = "#,##0.00";
    });
    setBottomBorder(sheet, row, 1, 7);
  });
  sheet.autoFilter = { from: "A3", to: `G${result.tokenSweepSeries.length + 3}` };
  sheet.views = [{ state: "frozen", ySplit: 3, showGridLines: false }];
  applyBaseSheetStyle(sheet);
  return sheet;
}

export async function buildPerformanceWorkbook({
  model,
  snapshot,
  result,
  exportedAt = new Date()
}: PerformanceExcelExportInput) {
  const workbook = new ExcelJS.Workbook();
  workbook.creator = "LLM Perf Calculator";
  workbook.lastModifiedBy = "LLM Perf Calculator";
  workbook.created = exportedAt;
  workbook.modified = exportedAt;
  workbook.subject = "LLM performance estimation result and formula trace";
  workbook.title = `${model.displayName} Performance Estimate`;
  workbook.description =
    "Reference-style grouped estimate detail with assumptions, results, memory breakdown, formula trace, and token trend.";
  workbook.calcProperties.fullCalcOnLoad = true;

  addEstimateDetailSheet(workbook, model, snapshot, result, exportedAt);
  addPrefillProjectionSheet(workbook, model, result);
  addDecodeProjectionSheet(workbook, model, snapshot, result);
  addResultsSheet(workbook, result);
  addFormulaTraceSheet(workbook, result);
  addTrendSheet(workbook, result);

  return workbook;
}

export async function exportPerformanceWorkbook(input: PerformanceExcelExportInput) {
  const workbook = await buildPerformanceWorkbook(input);
  const buffer = await workbook.xlsx.writeBuffer();
  const blob = new Blob([new Uint8Array(buffer)], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  });
  const date = (input.exportedAt ?? new Date()).toISOString().slice(0, 10);
  const filename = `llm-perf-${safeFilePart(input.model.displayName)}-${date}.xlsx`;
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
