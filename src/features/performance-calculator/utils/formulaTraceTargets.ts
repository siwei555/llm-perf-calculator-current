import type { FormulaTraceSection } from "../../../domain/performance/types";

export const formulaSectionTargets: Record<FormulaTraceSection["category"], string> = {
  prefill: "prefill-flops",
  decode: "decode-tps",
  memory: "decode-memory"
};

export function getFormulaTraceRowTarget(
  category: FormulaTraceSection["category"],
  rowIndex: number
) {
  return `formula-trace-${category}-${rowIndex}`;
}
