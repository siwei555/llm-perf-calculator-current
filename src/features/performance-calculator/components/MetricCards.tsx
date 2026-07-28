import type { PerformanceSummary } from "../../../domain/performance/types";

type Props = {
  summary: PerformanceSummary;
};

const cards = [
  {
    key: "ttftMs",
    label: "TTFT",
    unit: "s",
    subtext: "Prefill completion latency"
  },
  {
    key: "prefillTps",
    label: "Prefill TPS",
    unit: "tokens/s",
    subtext: "Current platform estimate"
  },
  {
    key: "decodeTps",
    label: "Decode TPS",
    unit: "tokens/s",
    subtext: "Steady-state decode rate"
  },
  {
    key: "totalRuntimeMemoryGb",
    label: "Runtime Memory",
    unit: "GB",
    subtext: "Weights + cache + temp"
  }
] as const;

function formatFriendlyDuration(totalSeconds: number) {
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(2)}s`;
  }

  const roundedSeconds = Math.round(totalSeconds);
  const hours = Math.floor(roundedSeconds / 3600);
  const minutes = Math.floor((roundedSeconds % 3600) / 60);
  const seconds = roundedSeconds % 60;
  const parts = [];

  if (hours > 0) {
    parts.push(`${hours}h`);
  }

  if (minutes > 0 || hours > 0) {
    parts.push(`${minutes}m`);
  }

  parts.push(`${seconds}s`);

  return parts.join(" ");
}

export function MetricCards({ summary }: Props) {
  return (
    <div className="metric-block">
      <div className="metric-grid">
        {cards.map((card) => {
          const rawValue = summary[card.key];
          const isTtft = card.key === "ttftMs";
          const displayValue = isTtft ? rawValue / 1000 : rawValue;
          const tag =
            card.key === "prefillTps"
              ? summary.prefillBottleneck
              : card.key === "decodeTps"
                ? summary.decodeBottleneck
                : summary.memoryFitsCapacity
                  ? "fits"
                  : "insufficient";

          return (
            <article key={card.key} className="metric-card">
              <p className="metric-card__label">{card.label}</p>
              <strong className="metric-card__value">
                {displayValue.toFixed(isTtft ? 2 : 1)}
              </strong>
              <span className="metric-card__unit">{card.unit}</span>
              {isTtft ? (
                <p className="metric-card__secondary">
                  {formatFriendlyDuration(displayValue)}
                </p>
              ) : null}
              <p className="metric-card__subtext">{card.subtext}</p>
              <span className="metric-card__tag">{tag}</span>
            </article>
          );
        })}
      </div>
      {!summary.memoryFitsCapacity ? (
        <p className="memory-warning">
          内存不足：估算需要 {summary.totalRuntimeMemoryGb.toFixed(2)} GB，当前容量不足。
        </p>
      ) : null}
    </div>
  );
}
