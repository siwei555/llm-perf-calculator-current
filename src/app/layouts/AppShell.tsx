import {
  useCallback,
  useLayoutEffect,
  useRef,
  type PropsWithChildren,
  type UIEvent
} from "react";
import { useLocation } from "react-router-dom";
import { SidebarNav } from "../../components/layout/SidebarNav";
import { useCalculatorContext } from "../../features/performance-calculator/state/CalculatorProvider";

const pageScrollPositions = new Map<string, number>();
const calculationResetPaths = ["/model-structure", "/formula-notes"];

export function AppShell({ children }: PropsWithChildren) {
  const location = useLocation();
  const { calculationRevision } = useCalculatorContext();
  const mainRef = useRef<HTMLDivElement>(null);
  const activePathRef = useRef(location.pathname);

  const handleMainScroll = useCallback((event: UIEvent<HTMLDivElement>) => {
    pageScrollPositions.set(
      activePathRef.current,
      event.currentTarget.scrollTop
    );
  }, []);

  useLayoutEffect(() => {
    const scroller = mainRef.current;
    if (!scroller) {
      return;
    }

    activePathRef.current = location.pathname;
    scroller.scrollTop = pageScrollPositions.get(location.pathname) ?? 0;
  }, [location.pathname]);

  useLayoutEffect(() => {
    calculationResetPaths.forEach((path) => {
      pageScrollPositions.set(path, 0);
    });
  }, [calculationRevision]);

  return (
    <div className="app-shell">
      <SidebarNav />
      <div
        className="app-shell__main"
        ref={mainRef}
        onScroll={handleMainScroll}
      >
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
