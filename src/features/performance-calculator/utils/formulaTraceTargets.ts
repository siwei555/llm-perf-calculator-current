import type { FormulaTraceSection } from "../../../domain/performance/types";

export const formulaSectionTargets: Record<FormulaTraceSection["category"], string> = {
  prefill: "prefill-flops",
  decode: "decode-tps",
  memory: "decode-memory"
};

export function groupFormulaTraceSections(sections: FormulaTraceSection[]): FormulaTraceSection[] {
  return sections.reduce<FormulaTraceSection[]>((groups, section) => {
    const group = groups.find((item) => item.category === section.category);
    if (group) {
      group.rows.push(...section.rows);
    } else {
      groups.push({ category: section.category, rows: [...section.rows] });
    }
    return groups;
  }, []);
}

export function getFormulaTraceRowTarget(
  category: FormulaTraceSection["category"],
  rowIndex: number
) {
  return `formula-trace-${category}-${rowIndex}`;
}
