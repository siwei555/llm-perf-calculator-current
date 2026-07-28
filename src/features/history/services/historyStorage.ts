import type { CalculationHistoryRecord } from "../../../domain/history/types";

const HISTORY_STORAGE_KEY = "llm-perf-calculator:calculation-history:v1";

function hasLocalStorage() {
  return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
}

export function loadHistoryRecords(): CalculationHistoryRecord[] {
  if (!hasLocalStorage()) {
    return [];
  }

  try {
    const serialized = window.localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!serialized) {
      return [];
    }

    const parsed: unknown = JSON.parse(serialized);
    return Array.isArray(parsed) ? (parsed as CalculationHistoryRecord[]) : [];
  } catch {
    return [];
  }
}

export function saveHistoryRecords(records: CalculationHistoryRecord[]) {
  if (!hasLocalStorage()) {
    return;
  }

  try {
    window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(records));
  } catch {
    // Storage may be unavailable or full. Keep the in-memory history usable.
  }
}

export function clearStoredHistoryRecords() {
  if (!hasLocalStorage()) {
    return;
  }

  try {
    window.localStorage.removeItem(HISTORY_STORAGE_KEY);
  } catch {
    // Keep clear-history usable in memory even when storage access fails.
  }
}
