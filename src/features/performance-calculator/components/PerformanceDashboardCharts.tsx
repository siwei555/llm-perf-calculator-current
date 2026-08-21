import { useState } from "react";
import type { ComparisonResult } from "../types/comparison";
import { getResourceScaling } from "../services/platformScaling";

type ScaleMode = "linear" | "log";
type MemoryMode = "total" | "weights" | "cache" | "temp" | "overhead";
type MemoryScope = "aggregate" | "per-chip";
type Metric = "prefillTps" | "decodeTps" | "ttftMs";

const WIDTH = 720;
const HEIGHT = 330;
const PAD = { left: 72, right: 20, top: 20, bottom: 48 };

function shortNumber(value: number) {
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(value >= 10_000 ? 0 : 1)}K`;
  return value.toFixed(value < 10 ? 2 : 0);
}

function domain(values: number[]) {
  const finiteValues = values.filter(Number.isFinite);
  const max = Math.max(...finiteValues);
  if (!Number.isFinite(max) || max <= 0) return [0, 1] as const;
  return [0, max * 1.08] as const;
}

function metricName(metric: Metric) {
  if (metric === "prefillTps") return "Prefill TPS";
  if (metric === "decodeTps") return "Initial Decode TPS";
  return "TTFT";
}

function metricUnit(metric: Metric) {
  if (metric === "prefillTps" || metric === "decodeTps") return "tokens/s";
  return "ms";
}

function MultiLineChart({ results, metric, scaleMode }: { results: ComparisonResult[]; metric: Metric; scaleMode: ScaleMode }) {
  const seriesFor = (result: ComparisonResult) => scaleMode === "log"
    ? result.logarithmicTokenSweepSeries
    : result.tokenSweepSeries;
  const points = results.flatMap(seriesFor);
  if (points.length === 0) return <p className="empty-state">请先开始对比。</p>;
  const tokens = points.map((point) => point.tokenLength);
  const logarithmicTokens = Array.from(new Set(tokens.filter((token) => token > 0))).sort((a, b) => a - b);
  const xMin = scaleMode === "log" ? logarithmicTokens[0] : 0;
  const xMax = Math.max(...tokens);
  const visiblePoints = points.filter((point) => Number.isFinite(point[metric]));
  const [yMin, yMax] = domain(visiblePoints.map((point) => point[metric]));
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const xValue = (value: number) => scaleMode === "log" ? Math.log(Math.max(xMin, value)) : value;
  const x0 = xValue(xMin);
  const xRange = Math.max(xValue(xMax) - x0, 1);
  const yRange = Math.max(yMax - yMin, 1e-9);
  const x = (value: number) => PAD.left + ((xValue(value) - x0) / xRange) * plotW;
  const y = (value: number) => PAD.top + plotH - ((value - yMin) / yRange) * plotH;
  const yTicks = Array.from({ length: 5 }, (_, index) => yMin + ((yMax - yMin) * index) / 4);
  const logarithmicTickStep = Math.max(1, Math.ceil(logarithmicTokens.length / 11));
  const xTicks = scaleMode === "log"
    ? logarithmicTokens.filter((_, index) => index % logarithmicTickStep === 0 || index === logarithmicTokens.length - 1)
    : Array.from(new Set([0, ...tokens.filter((_, index) => index % Math.max(1, Math.ceil(tokens.length / 4)) === 0), xMax])).slice(0, 6);

  return (
    <>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="dashboard-svg" role="img">
        {yTicks.map((tick) => <g key={tick}><line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(tick)} y2={y(tick)} className="dashboard-grid-line" /><text x={PAD.left - 10} y={y(tick) + 4} textAnchor="end">{shortNumber(tick)}</text></g>)}
        {xTicks.map((tick) => <g key={tick}><line x1={x(tick)} x2={x(tick)} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="dashboard-grid-line" /><text x={x(tick)} y={HEIGHT - 18} textAnchor="middle">{shortNumber(tick)}</text></g>)}
        {results.map((result) => {
          const { profile } = result;
          const visibleSeries = seriesFor(result).filter((point) => Number.isFinite(point[metric]));
          const path = visibleSeries.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.tokenLength)},${y(point[metric])}`).join(" ");
          return <g key={profile.id}><path d={path} fill="none" stroke={profile.color} strokeWidth="2.5" />{visibleSeries.map((point) => {
            const phaseDetails = metric === "prefillTps"
              ? `\nPrefill compute scaling: ${getResourceScaling(profile.platform, "prefill", "compute").toFixed(2)}x\nPrefill bandwidth scaling: ${getResourceScaling(profile.platform, "prefill", "bandwidth").toFixed(2)}x\nBottleneck: ${point.prefillBottleneck}`
              : metric === "decodeTps"
                ? `\nDecode compute scaling: ${getResourceScaling(profile.platform, "decode", "compute").toFixed(2)}x\nDecode bandwidth scaling: ${getResourceScaling(profile.platform, "decode", "bandwidth").toFixed(2)}x\nBottleneck: ${point.decodeBottleneck}`
                : "";
            return <circle key={point.tokenLength} cx={x(point.tokenLength)} cy={y(point[metric])} r="3" fill={profile.color}><title>{`${profile.label}\nContext: ${point.tokenLength.toLocaleString()}\n${metricName(metric)}: ${point[metric].toFixed(2)} ${metricUnit(metric)}${phaseDetails}`}</title></circle>;
          })}</g>;
        })}
        <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="dashboard-axis" />
        <line x1={PAD.left} x2={WIDTH - PAD.right} y1={HEIGHT - PAD.bottom} y2={HEIGHT - PAD.bottom} className="dashboard-axis" />
      </svg>
    </>
  );
}

function ProfileLegend({ results }: { results: ComparisonResult[] }) {
  return <div className="dashboard-legend">{results.map(({ profile }) => <span key={profile.id}><i style={{ background: profile.color }} />{profile.label}</span>)}</div>;
}

function MemoryUsageChart({ results, mode, scope, scaleMode }: { results: ComparisonResult[]; mode: MemoryMode; scope: MemoryScope; scaleMode: ScaleMode }) {
  const memorySeriesFor = (result: ComparisonResult) => scaleMode === "log"
    ? result.logarithmicTokenSweepSeries.map((point) => ({
        tokenLength: point.tokenLength,
        weightsGb: point.weightsGb,
        persistentCacheGb: point.persistentCacheGb,
        temporaryMemoryGb: point.temporaryMemoryGb,
        runtimeOverheadGb: point.runtimeOverheadGb,
        totalGb: point.totalRuntimeMemoryGb
      }))
    : result.memorySweepSeries;
  const allPoints = results.flatMap(memorySeriesFor);
  if (allPoints.length === 0) return <p className="empty-state">请先开始对比。</p>;
  const maxGroups = 10;
  const sampled = results.map((result) => {
    const memorySweepSeries = memorySeriesFor(result);
    return {
      ...result,
      memorySweepSeries: memorySweepSeries.filter((_, index, list) =>
        scaleMode === "log" || index % Math.max(1, Math.ceil(list.length / maxGroups)) === 0 || index === list.length - 1
      )
    };
  });
  const groupCount = Math.max(...sampled.map((result) => result.memorySweepSeries.length));
  const memoryComponents = (point: (typeof allPoints)[number], chipCount: number) => {
    const chips = Math.max(1, chipCount);
    const weights = scope === "per-chip" ? point.weightsGb / chips : point.weightsGb;
    const cache = point.persistentCacheGb;
    const temp = scope === "per-chip" ? point.temporaryMemoryGb / Math.min(chips, 2) : point.temporaryMemoryGb;
    const overhead = point.runtimeOverheadGb;
    return { weights, cache, temp, overhead, total: weights + cache + temp + overhead };
  };
  const valueGb = (point: (typeof allPoints)[number], chipCount: number) => {
    const components = memoryComponents(point, chipCount);
    if (mode === "weights") return components.weights;
    if (mode === "cache") return components.cache;
    if (mode === "temp") return components.temp;
    if (mode === "overhead") return components.overhead;
    return components.total;
  };
  const valueMb = (point: (typeof allPoints)[number], chipCount: number) => valueGb(point, chipCount) * 1000;
  const max = Math.max(...sampled.flatMap((result) => result.memorySweepSeries.map((point) => valueMb(point, result.profile.platform.chipCount))), 1) * 1.08;
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const groupW = plotW / Math.max(groupCount, 1);
  const barW = Math.max(3, Math.min(18, (groupW - 6) / Math.max(results.length, 1)));

  return <><svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="dashboard-svg dashboard-svg--memory" role="img">
    {Array.from({ length: 5 }, (_, index) => (max * index) / 4).map((tick) => { const yy = PAD.top + plotH - (tick / max) * plotH; return <g key={tick}><line x1={PAD.left} x2={WIDTH - PAD.right} y1={yy} y2={yy} className="dashboard-grid-line" /><text x={PAD.left - 10} y={yy + 4} textAnchor="end">{shortNumber(tick)}</text></g>; })}
    {sampled.map(({ profile, memorySweepSeries }, profileIndex) => memorySweepSeries.map((point, pointIndex) => {
      const xx = PAD.left + pointIndex * groupW + (groupW - barW * results.length) / 2 + profileIndex * barW;
      const components = memoryComponents(point, profile.platform.chipCount);
      const segments = mode === "total"
        ? [
            { key: "weights", value: components.weights * 1000, opacity: 1 },
            { key: "cache", value: components.cache * 1000, opacity: 0.62 },
            { key: "temp", value: components.temp * 1000, opacity: 0.36 },
            { key: "overhead", value: components.overhead * 1000, opacity: 0.2 }
          ]
        : [{
            key: mode,
            value: valueMb(point, profile.platform.chipCount),
            opacity: mode === "cache" ? 0.62 : mode === "temp" ? 0.36 : mode === "overhead" ? 0.2 : 1
          }];
      let accumulated = 0;
      return <g key={`${profile.id}-${point.tokenLength}`}>{segments.map((segment) => {
        const h = (segment.value / max) * plotH;
        const yy = PAD.top + plotH - ((accumulated + segment.value) / max) * plotH;
        accumulated += segment.value;
        return <rect key={segment.key} x={xx} y={yy} width={barW - 1} height={Math.max(h, 0)} fill={profile.color} fillOpacity={segment.opacity} rx={segment.key === segments[segments.length - 1].key ? 2 : 0} />;
      })}<title>{`${profile.label}\nContext: ${point.tokenLength.toLocaleString()}\n口径: ${scope === "per-chip" ? "单卡显存压力" : "总显存需求"}\nDisplayed total: ${(components.total * 1000).toFixed(1)} MB\nWeights: ${(components.weights * 1000).toFixed(1)} MB\nKV/State: ${(components.cache * 1000).toFixed(1)} MB\nTemp: ${(components.temp * 1000).toFixed(1)} MB\nOverhead: ${(components.overhead * 1000).toFixed(1)} MB`}</title></g>;
    }))}
    {sampled[0]?.memorySweepSeries.map((point, index) => <text key={point.tokenLength} x={PAD.left + index * groupW + groupW / 2} y={HEIGHT - 18} textAnchor="middle">{shortNumber(point.tokenLength)}</text>)}
    <line x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={HEIGHT - PAD.bottom} className="dashboard-axis" /><line x1={PAD.left} x2={WIDTH - PAD.right} y1={HEIGHT - PAD.bottom} y2={HEIGHT - PAD.bottom} className="dashboard-axis" />
  </svg><ProfileLegend results={results} />{mode === "total" ? <div className="memory-stack-legend"><span><i className="memory-stack-legend__weight" />Weight</span><span><i className="memory-stack-legend__cache" />KV/State</span><span><i className="memory-stack-legend__temp" />Temp</span><span><i className="memory-stack-legend__overhead" />Overhead</span></div> : null}</>;
}

export function PerformanceDashboardCharts({ results }: { results: ComparisonResult[] }) {
  const [scaleMode, setScaleMode] = useState<ScaleMode>("linear");
  const [memoryMode, setMemoryMode] = useState<MemoryMode>("total");
  const [memoryScope, setMemoryScope] = useState<MemoryScope>("aggregate");
  const [showTps, setShowTps] = useState(true);
  const [showTtft, setShowTtft] = useState(true);
  const showTrendLegend = showTps || showTtft;
  return <section className="dashboard-section">
    <div className="dashboard-heading"><div><h3>性能数据看板</h3><p>按需展示 Token 趋势；所有曲线共享当前 Token Sweep，Decode 表示 initial decode TPS。</p></div><div className="dashboard-controls"><div className="dashboard-metric-toggles"><label><input type="checkbox" checked={showTps} onChange={(event) => setShowTps(event.target.checked)} />Prefill &amp; Decode TPS</label><label><input type="checkbox" checked={showTtft} onChange={(event) => setShowTtft(event.target.checked)} />TTFT</label></div><label>X 轴<select value={scaleMode} onChange={(event) => setScaleMode(event.target.value as ScaleMode)}><option value="linear">线性</option><option value="log">对数</option></select></label></div></div>
    <div className="dashboard-grid">
      {showTps ? <><article className="panel dashboard-chart"><div className="dashboard-chart__title"><h4><i className="dashboard-title-dot dashboard-title-dot--prefill" />Prefill Speed</h4><span>单位 tps</span></div><MultiLineChart results={results} metric="prefillTps" scaleMode={scaleMode} /></article><article className="panel dashboard-chart"><div className="dashboard-chart__title"><h4><i className="dashboard-title-dot dashboard-title-dot--decode" />Decode Speed</h4><span>单位 tps</span></div><MultiLineChart results={results} metric="decodeTps" scaleMode={scaleMode} /></article></> : null}
      {showTtft ? <article className="panel dashboard-chart dashboard-chart--full"><div className="dashboard-chart__title"><h4><i className="dashboard-title-dot dashboard-title-dot--ttft" />TTFT</h4><span>单位 ms</span></div><MultiLineChart results={results} metric="ttftMs" scaleMode={scaleMode} /></article> : null}
      {showTrendLegend ? <div className="dashboard-shared-legend"><ProfileLegend results={results} /></div> : null}
      <article className="panel dashboard-chart dashboard-chart--memory"><div className="dashboard-chart__title"><h4><i className="dashboard-title-dot dashboard-title-dot--memory" />Memory Usage（{memoryScope === "per-chip" ? "单卡显存压力" : "总显存需求"}）</h4><div><span>单位 MB</span><select aria-label="显存口径" value={memoryScope} onChange={(event) => setMemoryScope(event.target.value as MemoryScope)}><option value="aggregate">总显存需求</option><option value="per-chip">单卡显存压力</option></select><select aria-label="显存组成项" value={memoryMode} onChange={(event) => setMemoryMode(event.target.value as MemoryMode)}><option value="total">全部</option><option value="weights">只看 Weight</option><option value="cache">只看 KV/State Cache</option><option value="temp">只看 Temp</option><option value="overhead">只看 Runtime Overhead</option></select></div></div><MemoryUsageChart results={results} mode={memoryMode} scope={memoryScope} scaleMode={scaleMode} /></article>
    </div>
  </section>;
}
