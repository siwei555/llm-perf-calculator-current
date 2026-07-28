import { Outlet } from "react-router-dom";
import { CalculatorProvider } from "../features/performance-calculator/state/CalculatorProvider";
import { AppShell } from "./layouts/AppShell";

export function App() {
  return (
    <CalculatorProvider>
      <AppShell>
        <Outlet />
      </AppShell>
    </CalculatorProvider>
  );
}
