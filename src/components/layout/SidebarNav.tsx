import { NavLink } from "react-router-dom";

const navItems = [
  {
    to: "/performance-calculator",
    label: "性能计算",
    meta: "Main"
  },
  {
    to: "/model-structure",
    label: "模型结构",
    meta: "Model"
  },
  {
    to: "/formula-notes",
    label: "公式说明",
    meta: "Formula"
  },
  {
    to: "/history",
    label: "历史记录",
    meta: "Logs"
  }
];

export function SidebarNav() {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__brand-mark">LP</div>
        <div>
          <p className="sidebar__brand-title">LLM Perf</p>
          <p className="sidebar__brand-subtitle">Calculator</p>
        </div>
      </div>
      <nav className="sidebar__nav" aria-label="Primary">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              isActive ? "sidebar__link sidebar__link--active" : "sidebar__link"
            }
          >
            <span>{item.label}</span>
            <span className="sidebar__meta">{item.meta}</span>
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
