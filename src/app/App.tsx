import { Navigate, useLocation } from "react-router-dom";
import { CalculatorProvider } from "../features/performance-calculator/state/CalculatorProvider";
import { FormulaNotesPage } from "../pages/formula-notes/FormulaNotesPage";
import { HistoryPage } from "../pages/history/HistoryPage";
import { ModelStructurePage } from "../pages/model-structure/ModelStructurePage";
import { PerformanceCalculatorPage } from "../pages/performance-calculator/PerformanceCalculatorPage";
import { AppShell } from "./layouts/AppShell";

const persistentPages = [
  {
    path: "/performance-calculator",
    component: PerformanceCalculatorPage
  },
  {
    path: "/model-structure",
    component: ModelStructurePage
  },
  {
    path: "/formula-notes",
    component: FormulaNotesPage
  },
  {
    path: "/history",
    component: HistoryPage
  }
] as const;

export function App() {
  const location = useLocation();

  if (location.pathname === "/") {
    return <Navigate to="/performance-calculator" replace />;
  }

  return (
    <CalculatorProvider>
      <AppShell>
        {persistentPages.map(({ path, component: Page }) => (
          <div
            key={path}
            className="route-view"
            hidden={location.pathname !== path}
            aria-hidden={location.pathname !== path}
          >
            <Page />
          </div>
        ))}
      </AppShell>
    </CalculatorProvider>
  );
}
