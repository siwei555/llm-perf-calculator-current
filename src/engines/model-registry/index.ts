import type { ModelDefinition, ModelId } from "../../domain/model/types";
import { deepseekV4Models } from "./deepseekV4Models";
import { gemma4Models } from "./gemma4Models";
import { qwen3_5Models } from "./qwen3_5Models";
import { qwen3_6Models } from "./qwen3_6Models";

export const modelRegistry: ModelDefinition[] = [
  ...deepseekV4Models,
  ...gemma4Models,
  ...qwen3_5Models,
  ...qwen3_6Models
];

const familyDisplayNames: Record<string, string> = {
  "deepseek-v4": "DeepSeek V4",
  "gemma-4": "Gemma 4",
  "qwen3.5": "Qwen3.5",
  "qwen3.6": "Qwen3.6"
};

export type ModelFamilyOption = {
  id: string;
  displayName: string;
};

export function getModelFamilies(): ModelFamilyOption[] {
  return Array.from(new Set(modelRegistry.map((entry) => entry.family))).map((family) => ({
    id: family,
    displayName: familyDisplayNames[family] ?? family
  }));
}

export function getModelsByFamily(family: string): ModelDefinition[] {
  return modelRegistry.filter((entry) => entry.family === family);
}

export function getModelDefinition(modelId: ModelId): ModelDefinition {
  const model = modelRegistry.find((entry) => entry.id === modelId);

  if (!model) {
    throw new Error(`Unknown model: ${modelId}`);
  }

  return model;
}
