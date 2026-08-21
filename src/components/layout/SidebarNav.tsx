import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

type SidebarSection = {
  id: string;
  label: string;
};

type SidebarItem = {
  to: string;
  label: string;
  meta: string;
  sections: SidebarSection[];
};

const navItems: SidebarItem[] = [
  {
    to: "/performance-calculator",
    label: "性能计算",
    meta: "Main",
    sections: [
      { id: "performance-model-selection", label: "模型选择" },
      { id: "performance-input-length", label: "输入长度" },
      { id: "performance-platform", label: "平台参数" },
      { id: "performance-assumptions", label: "计算假设" },
      { id: "performance-comparison-settings", label: "对比设置" },
      { id: "performance-overview", label: "核心结果" },
      { id: "performance-comparison", label: "性能对比" },
      { id: "performance-dashboard", label: "性能对比图" },
      { id: "performance-formula-trace", label: "公式追溯" },
      { id: "performance-intermediate", label: "中间量结果表" }
    ]
  },
  {
    to: "/model-structure",
    label: "模型结构",
    meta: "Model",
    sections: [
      { id: "structure-model-selection", label: "模型选择" },
      { id: "structure-flow", label: "结构流图" },
      { id: "structure-modules", label: "模块参数" },
      { id: "structure-schedule", label: "层级排布" },
      { id: "structure-parameter-links", label: "参数与性能关联" }
    ]
  },
  {
    to: "/formula-notes",
    label: "公式说明",
    meta: "Formula",
    sections: [
      { id: "formula-model-selection", label: "模型选择" },
      { id: "prefill-flops", label: "Prefill FLOPs" },
      { id: "prefill-tps", label: "Prefill TPS" },
      { id: "decode-tps", label: "Decode TPS" },
      { id: "decode-memory", label: "Decode Memory" }
    ]
  },
  {
    to: "/history",
    label: "历史记录",
    meta: "Logs",
    sections: [
      { id: "history-filters", label: "筛选与排序" },
      { id: "history-records", label: "计算记录" }
    ]
  }
];

type Props = {
  onSectionNavigate: (path: string, sectionId: string) => void;
};

export function SidebarNav({ onSectionNavigate }: Props) {
  const location = useLocation();
  const [openPaths, setOpenPaths] = useState<Set<string>>(
    () => new Set([location.pathname])
  );

  useEffect(() => {
    setOpenPaths((current) => {
      const next = new Set(current);
      next.add(location.pathname);
      return next;
    });
  }, [location.pathname]);

  const toggleGroup = (path: string) => {
    setOpenPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  };

  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__brand-mark">LP</div>
        <div>
          <p className="sidebar__brand-title">LLM Perf</p>
          <p className="sidebar__brand-subtitle">Calculator</p>
        </div>
      </div>
      <nav className="sidebar__nav" aria-label="主导航">
        {navItems.map((item) => {
          const isOpen = openPaths.has(item.to);
          const groupId = `sidebar-sections-${item.meta.toLowerCase()}`;

          return (
            <div className="sidebar__group" key={item.to}>
              <div className="sidebar__group-heading">
                <NavLink
                  to={item.to}
                  className={({ isActive }) =>
                    isActive ? "sidebar__link sidebar__link--active" : "sidebar__link"
                  }
                >
                  <span>{item.label}</span>
                  <span className="sidebar__meta">{item.meta}</span>
                </NavLink>
                <button
                  type="button"
                  className="sidebar__toggle"
                  aria-label={`${isOpen ? "收起" : "展开"}${item.label}小标题`}
                  aria-expanded={isOpen}
                  aria-controls={groupId}
                  onClick={() => toggleGroup(item.to)}
                >
                  <span aria-hidden="true" />
                </button>
              </div>
              <div
                id={groupId}
                className={isOpen ? "sidebar__sections sidebar__sections--open" : "sidebar__sections"}
              >
                <div className="sidebar__sections-inner">
                  {item.sections.map((section) => (
                    <button
                      type="button"
                      className="sidebar__section-link"
                      key={section.id}
                      onClick={() => onSectionNavigate(item.to, section.id)}
                    >
                      <span aria-hidden="true" />
                      {section.label}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </nav>
    </aside>
  );
}
