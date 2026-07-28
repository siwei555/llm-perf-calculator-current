import type {
  TokenSweepPoint,
  TrendMetricKey
} from "../../../domain/performance/types";

type Props = {
  points: TokenSweepPoint[];
  selectedMetric: TrendMetricKey;
  onMetricChange: (metric: TrendMetricKey) => void;
  showDataPoints: boolean;
  onShowDataPointsChange: (value: boolean) => void;
  showBottleneckBackground: boolean;
  onShowBottleneckBackgroundChange: (value: boolean) => void;
};

const metricOptions: { key: TrendMetricKey | "prefillDecodeTps"; label: string }[] = [
  { key: "prefillDecodeTps", label: "Prefill & Decode TPS" },
  { key: "ttftMs", label: "TTFT" },
  { key: "totalRuntimeMemoryGb", label: "Total Runtime Memory" }
];

const CHART_WIDTH = 760;
const CHART_HEIGHT = 200;
const PLOT = { left: 72, right: 24, top: 16, bottom: 42 };

function plotWidth() { return CHART_WIDTH - PLOT.left - PLOT.right; }
function plotHeight() { return CHART_HEIGHT - PLOT.top - PLOT.bottom; }

function formatTokenLength(value: number) {
  if (value >= 1024 * 1024) {
    return `${(value / 1024 / 1024).toFixed(value % (1024 * 1024) === 0 ? 0 : 1)}M`;
  }
  if (value >= 1024) {
    return `${(value / 1024).toFixed(value % 1024 === 0 ? 0 : 1)}K`;
  }
  return String(value);
}

function formatMetricValue(metric: TrendMetricKey, value: number) {
  if (metric === "ttftMs") return `${(value / 1000).toFixed(2)} s`;
  if (metric === "totalRuntimeMemoryGb") return `${value.toFixed(2)} GB`;
  return `${value.toFixed(2)} tokens/s`;
}

function formatAxisTick(metric: TrendMetricKey, value: number) {
  if (metric === "ttftMs") return (value / 1000).toFixed(1);
  if (metric === "totalRuntimeMemoryGb") return value.toFixed(1);
  if (value >= 1000) return `${(value / 1000).toFixed(0)}K`;
  return value.toFixed(0);
}

function getMetricUnit(metric: TrendMetricKey) {
  if (metric === "ttftMs") return "s";
  if (metric === "totalRuntimeMemoryGb") return "GB";
  return "tokens/s";
}

function getMetricLabel(metric: TrendMetricKey) {
  return metricOptions.find((option) => option.key === metric)?.label ?? metric;
}

function getYDomain(values: number[]) {
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (minValue === maxValue) {
    const pad = Math.max(Math.abs(minValue) * 0.1, 1);
    return [minValue - pad, maxValue + pad] as const;
  }
  const pad = (maxValue - minValue) * 0.08;
  return [Math.max(0, minValue - pad), maxValue + pad] as const;
}

function buildPath(points: { x: number; y: number }[]) {
  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(" ");
}

function makeTicks(min: number, max: number, count: number) {
  if (count <= 1) return [min];
  return Array.from({ length: count }, (_, i) => min + ((max - min) * i) / (count - 1));
}

type SingleChartProps = {
  metric: TrendMetricKey;
  points: TokenSweepPoint[];
  color: string;
  showDataPoints: boolean;
  showBottleneckBackground: boolean;
  yDomainOverride?: readonly [number, number];
};

function SingleTrendChart({
  metric,
  points,
  color,
  showDataPoints,
  showBottleneckBackground,
  yDomainOverride
}: SingleChartProps) {
  const values = points.map((p) => p[metric]);
  const [yMin, yMax] = yDomainOverride ?? getYDomain(values);
  const xMin = points[0]?.tokenLength ?? 0;
  const xMax = points[points.length - 1]?.tokenLength ?? xMin + 1;
  const xRange = Math.max(xMax - xMin, 1);
  const yRange = Math.max(yMax - yMin, 1);
  const xScale = (t: number) => PLOT.left + ((t - xMin) / xRange) * plotWidth();
  const yScale = (v: number) => PLOT.top + plotHeight() - ((v - yMin) / yRange) * plotHeight();

  const plotPoints = points.map((p) => ({ x: xScale(p.tokenLength), y: yScale(p[metric]) }));
  const pathD = buildPath(plotPoints);
  const xTicks = makeTicks(xMin, xMax, 5);
  const yTicks = makeTicks(yMin, yMax, 4);

  return (
    <div className="trend-chart">
      <svg viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={`${getMetricLabel(metric)} trend`}>
        {showBottleneckBackground && (
          <>
            <rect x={PLOT.left} y={PLOT.top} width={plotWidth() / 2} height={plotHeight()} fill="rgba(37, 99, 235, 0.05)" />
            <rect x={PLOT.left + plotWidth() / 2} y={PLOT.top} width={plotWidth() / 2} height={plotHeight()} fill="rgba(14, 116, 144, 0.05)" />
          </>
        )}

        {/* Y-axis grid + ticks */}
        {yTicks.map((tick) => {
          const y = yScale(tick);
          return (
            <g key={`y-${tick}`}>
              <line x1={PLOT.left} x2={PLOT.left + plotWidth()} y1={y} y2={y} stroke="rgba(15, 23, 42, 0.08)" />
              <text x={PLOT.left - 10} y={y + 4} textAnchor="end" className="chart-axis-text">
                {formatAxisTick(metric, tick)}
              </text>
            </g>
          );
        })}

        {/* X-axis ticks */}
        {xTicks.map((tick) => {
          const x = xScale(tick);
          return (
            <g key={`x-${tick}`}>
              <line x1={x} x2={x} y1={PLOT.top} y2={PLOT.top + plotHeight()} stroke="rgba(15, 23, 42, 0.05)" />
              <text x={x} y={PLOT.top + plotHeight() + 20} textAnchor="middle" className="chart-axis-text">
                {formatTokenLength(tick)}
              </text>
            </g>
          );
        })}

        {/* Axis lines */}
        <line x1={PLOT.left} x2={PLOT.left} y1={PLOT.top} y2={PLOT.top + plotHeight()} stroke="rgba(15, 23, 42, 0.36)" />
        <line x1={PLOT.left} x2={PLOT.left + plotWidth()} y1={PLOT.top + plotHeight()} y2={PLOT.top + plotHeight()} stroke="rgba(15, 23, 42, 0.36)" />

        {/* X-axis label */}
        <text x={PLOT.left + plotWidth() / 2} y={CHART_HEIGHT - 6} textAnchor="middle" className="chart-axis-label">
          Token Length (tokens)
        </text>

        {/* Y-axis label */}
        <text x={14} y={PLOT.top + plotHeight() / 2} textAnchor="middle" className="chart-axis-label" transform={`rotate(-90 14 ${PLOT.top + plotHeight() / 2})`}>
          {getMetricLabel(metric)} ({getMetricUnit(metric)})
        </text>

        {/* Trend line */}
        <path d={pathD} fill="none" stroke={color} strokeWidth="2.5" />

        {/* Data points */}
        {showDataPoints &&
          points.map((point) => {
            const x = xScale(point.tokenLength);
            const y = yScale(point[metric]);
            return (
              <g key={point.tokenLength} className="chart-point">
                <circle cx={x} cy={y} r="3.5" fill={color} />
                <circle cx={x} cy={y} r="10" fill="transparent" />
                <title>
                  {`Token Length: ${point.tokenLength.toLocaleString()}
${getMetricLabel(metric)}: ${formatMetricValue(metric, point[metric])}
Prefill TPS: ${formatMetricValue("prefillTps", point.prefillTps)}
Decode TPS: ${formatMetricValue("decodeTps", point.decodeTps)}
TTFT: ${formatMetricValue("ttftMs", point.ttftMs)}
Memory: ${formatMetricValue("totalRuntimeMemoryGb", point.totalRuntimeMemoryGb)}
Prefill Bottleneck: ${point.prefillBottleneck}
Decode Bottleneck: ${point.decodeBottleneck}`}
                </title>
              </g>
            );
          })}
      </svg>
    </div>
  );
}

export function TrendChart({
  points,
  selectedMetric,
  onMetricChange,
  showDataPoints,
  onShowDataPointsChange,
  showBottleneckBackground,
  onShowBottleneckBackgroundChange
}: Props) {
  const lastPoint = points.length > 0 ? points[points.length - 1] : undefined;
  const isTpsMode = selectedMetric === "prefillTps" || selectedMetric === "decodeTps";

  function handleDropdownChange(value: string) {
    if (value === "prefillDecodeTps") {
      // default to showing both TPS charts via prefillTps key
      onMetricChange("prefillTps");
    } else {
      onMetricChange(value as TrendMetricKey);
    }
  }

  // dropdown shows combined key when in TPS mode, actual metric otherwise
  const dropdownValue = isTpsMode ? "prefillDecodeTps" : selectedMetric;

  return (
    <article className="panel panel--large">
      <div className="chart-header">
        <div>
          <h3>Token 趋势图</h3>
          <p>按 token sweep 范围重新计算每个离散点。</p>
        </div>
        <div className="chart-controls">
          <select
            value={dropdownValue}
            onChange={(event) => handleDropdownChange(event.target.value)}
          >
            {metricOptions.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={showBottleneckBackground}
              onChange={(event) => onShowBottleneckBackgroundChange(event.target.checked)}
            />
            <span>Bottleneck</span>
          </label>
          <label className="inline-check">
            <input
              type="checkbox"
              checked={showDataPoints}
              onChange={(event) => onShowDataPointsChange(event.target.checked)}
            />
            <span>Points</span>
          </label>
        </div>
      </div>

      {isTpsMode ? (
        <>
          <SingleTrendChart
            metric="prefillTps"
            points={points}
            color="#2563eb"
            showDataPoints={showDataPoints}
            showBottleneckBackground={showBottleneckBackground}
          />
          <div style={{ height: 12 }} />
          <SingleTrendChart
            metric="decodeTps"
            points={points}
            color="#0f766e"
            showDataPoints={showDataPoints}
            showBottleneckBackground={showBottleneckBackground}
          />
        </>
      ) : (
        <SingleTrendChart
          metric={selectedMetric}
          points={points}
          color="#2563eb"
          showDataPoints={showDataPoints}
          showBottleneckBackground={showBottleneckBackground}
        />
      )}

      <div className="trend-footer">
        <span>
          Range: {formatTokenLength(points[0]?.tokenLength ?? 0)} - {formatTokenLength(lastPoint?.tokenLength ?? 0)}
        </span>
        <span>Points: {points.length}</span>
        {isTpsMode ? (
          <span className="chart-legend">
            <i className="chart-legend__primary" /> Prefill TPS
            <i className="chart-legend__secondary" /> Decode TPS
          </span>
        ) : (
          <span className="chart-legend">
            <i className="chart-legend__primary" /> {getMetricLabel(selectedMetric)}
          </span>
        )}
      </div>

      <div className="trend-data-table-wrap">
        <table className="data-table data-table--compact trend-data-table">
          <thead>
            <tr>
              <th scope="col">Token Length</th>
              <th scope="col">Prefill TPS</th>
              <th scope="col">Decode TPS</th>
              <th scope="col">TTFT</th>
              <th scope="col">Runtime Memory</th>
              <th scope="col">Bottleneck</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point) => (
              <tr key={point.tokenLength}>
                <td>{point.tokenLength.toLocaleString()}</td>
                <td>{formatMetricValue("prefillTps", point.prefillTps)}</td>
                <td>{formatMetricValue("decodeTps", point.decodeTps)}</td>
                <td>{formatMetricValue("ttftMs", point.ttftMs)}</td>
                <td>{formatMetricValue("totalRuntimeMemoryGb", point.totalRuntimeMemoryGb)}</td>
                <td>
                  Prefill: {point.prefillBottleneck}<br />
                  Decode: {point.decodeBottleneck}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}
