import type { IntermediateMetric } from "../../../domain/performance/types";

export function IntermediateMetricsTable({
  rows
}: {
  rows: IntermediateMetric[];
}) {
  return (
    <article className="panel panel--large">
      <h3>中间量结果表</h3>
      <div className="table-scroll">
        <table className="data-table data-table--intermediate">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Symbol</th>
              <th>Value</th>
              <th>Unit</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <td>{row.label}</td>
                <td>{row.symbol}</td>
                <td>{row.value}</td>
                <td>{row.unit}</td>
                <td>{row.source}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}
