import { createContext, useContext, type PropsWithChildren } from "react";
import { useCalculatorState } from "./useCalculatorState";

type CalculatorContextValue = ReturnType<typeof useCalculatorState>;

const CalculatorContext = createContext<CalculatorContextValue | null>(null);

export function CalculatorProvider({ children }: PropsWithChildren) {
  const calculator = useCalculatorState();

  return (
    <CalculatorContext.Provider value={calculator}>
      {children}
    </CalculatorContext.Provider>
  );
}

export function useCalculatorContext() {
  const value = useContext(CalculatorContext);

  if (!value) {
    throw new Error("useCalculatorContext must be used within CalculatorProvider");
  }

  return value;
}

