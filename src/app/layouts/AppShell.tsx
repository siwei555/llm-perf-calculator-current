import type { PropsWithChildren } from "react";
import { SidebarNav } from "../../components/layout/SidebarNav";

export function AppShell({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <SidebarNav />
      <div className="app-shell__main">
        <header className="topbar">
          <div>
            <p className="eyebrow">Engineering Workbench</p>
            <h1>LLM Perf Calculator</h1>
          </div>
        </header>
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}
