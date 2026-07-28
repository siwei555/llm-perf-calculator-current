import type { FormulaTraceSection } from "../../../domain/performance/types";

export function FormulaTraceCard({
  sections
}: {
  sections: FormulaTraceSection[];
}) {
  return (
    <article className="panel">
      <h3>公式追溯</h3>
      <div className="formula-trace">
        {sections.map((section) => (
          <details key={section.category} className="formula-trace__section">
            <summary className="formula-trace__summary">
              <span className="eyebrow">{section.category}</span>
              <span className="formula-trace__chevron" aria-hidden="true" />
            </summary>
            <div className="formula-trace__grid">
              {section.rows.map((row) => (
                <div key={row.label} className="formula-trace__row">
                  <strong>{row.label}</strong>
                  <code>{row.expression}</code>
                  <span>{row.evaluated}</span>
                  {row.sourceUrl ? (
                    <a
                      className="formula-trace__source"
                      href={row.sourceUrl}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {row.sourceLabel ?? "来源"} ↗
                    </a>
                  ) : null}
                </div>
              ))}
            </div>
          </details>
        ))}
      </div>
    </article>
  );
}
