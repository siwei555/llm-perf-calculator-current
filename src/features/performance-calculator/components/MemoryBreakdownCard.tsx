import type { MemoryBreakdownRow } from "../../../domain/performance/types";

type Props = {
  rows: MemoryBreakdownRow[];
  runtimeOverheadGb: number;
  runtimeOverheadError?: string;
  onRuntimeOverheadChange: (value: number) => void;
};

export function MemoryBreakdownCard({
  rows,
  runtimeOverheadGb,
  runtimeOverheadError,
  onRuntimeOverheadChange
}: Props) {
  return (
    <article className="panel">
      <h3>内存拆解</h3>
      <div className="stack-list">
        {rows.map((row) => {
          const isRuntimeOverhead = row.key === "runtimeOverhead";

          return (
            <div key={row.key} className="stack-list__row">
              <div className="memory-row__label">
                <strong>{row.label}</strong>
                {isRuntimeOverhead ? <small>估算假设，可手动编辑</small> : null}
              </div>
              {isRuntimeOverhead ? (
                <div className="memory-row__editor">
                  <input
                    type="number"
                    min="0"
                    step="0.1"
                    aria-label="Runtime Overhead（GB，估算假设）"
                    value={runtimeOverheadGb}
                    onChange={(event) => onRuntimeOverheadChange(Number(event.target.value))}
                  />
                  <span>GB</span>
                  {runtimeOverheadError ? (
                    <small className="field-error">{runtimeOverheadError}</small>
                  ) : null}
                </div>
              ) : (
                <span>{row.valueGb.toFixed(2)} GB</span>
              )}
            </div>
          );
        })}
      </div>
    </article>
  );
}
