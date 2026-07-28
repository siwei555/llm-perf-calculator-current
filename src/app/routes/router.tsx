import { createHashRouter, Navigate } from "react-router-dom";
import { App } from "../App";
import { FormulaNotesPage } from "../../pages/formula-notes/FormulaNotesPage";
import { HistoryPage } from "../../pages/history/HistoryPage";
import { ModelStructurePage } from "../../pages/model-structure/ModelStructurePage";
import { PerformanceCalculatorPage } from "../../pages/performance-calculator/PerformanceCalculatorPage";

export const appRouter = createHashRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true,
        element: <Navigate to="/performance-calculator" replace />
      },
      {
        path: "performance-calculator",
        element: <PerformanceCalculatorPage />
      },
      {
        path: "model-structure",
        element: <ModelStructurePage />
      },
      {
        path: "formula-notes",
        element: <FormulaNotesPage />
      },
      {
        path: "history",
        element: <HistoryPage />
      }
    ]
  }
]);
