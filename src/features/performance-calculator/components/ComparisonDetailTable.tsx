import type { ComparisonResult } from "../types/comparison";
import { getResourceScaling } from "../services/platformScaling";

function perChipMemoryGb(point: ComparisonResult["tokenSweepSeries"][number], chipCount: number) {
  const chips = Math.max(1, chipCount);
  return point.weightsGb / chips + point.persistentCacheGb + point.temporaryMemoryGb / Math.min(chips, 2) + point.runtimeOverheadGb;
}

export function ComparisonDetailTable({ results }: { results: ComparisonResult[] }) {
  if (results.length === 0) return null;
  return <article className="panel panel--large">
    <h3>对比明细</h3>
    <div className="table-scroll"><table className="data-table"><thead><tr><th>配置</th><th>Context</th><th>实际扩展倍率 P-C/P-B/D-C/D-B</th><th>Prefill TPS</th><th>Initial Decode TPS</th><th>总显存需求</th><th>单卡显存压力</th><th>Bottleneck</th></tr></thead><tbody>
      {results.flatMap(({ profile, tokenSweepSeries }) => tokenSweepSeries.map((point) => <tr key={`${profile.id}-${point.tokenLength}`}><td><span className="profile-dot" style={{ background: profile.color }} />{profile.label}</td><td>{point.tokenLength.toLocaleString()}</td><td>{getResourceScaling(profile.platform, "prefill", "compute").toFixed(2)}× / {getResourceScaling(profile.platform, "prefill", "bandwidth").toFixed(2)}× / {getResourceScaling(profile.platform, "decode", "compute").toFixed(2)}× / {getResourceScaling(profile.platform, "decode", "bandwidth").toFixed(2)}×</td><td>{point.prefillTps.toFixed(2)}</td><td>{point.decodeTps.toFixed(2)}</td><td>{point.totalRuntimeMemoryGb.toFixed(3)} GB</td><td>{perChipMemoryGb(point, profile.platform.chipCount).toFixed(3)} GB</td><td>Prefill: {point.prefillBottleneck}<br />Decode: {point.decodeBottleneck}</td></tr>))}
    </tbody></table></div>
  </article>;
}
