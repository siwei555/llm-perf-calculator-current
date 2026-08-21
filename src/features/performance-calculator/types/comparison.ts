import type { ModelId } from "../../../domain/model/types";
import type { MemorySweepPoint, TokenSweepPoint } from "../../../domain/performance/types";
import type { PlatformInput } from "../../../domain/platform/types";

export type PrecisionPresetId = "w4a8" | "w8a8" | "fp8" | "bf16" | "custom";

export type ComparisonProfile = {
  id: string;
  label: string;
  modelId: ModelId;
  platform: PlatformInput;
  precision: PrecisionPresetId;
  color: string;
  enabled: boolean;
};

export type ComparisonResult = {
  profile: ComparisonProfile;
  tokenSweepSeries: TokenSweepPoint[];
  logarithmicTokenSweepSeries: TokenSweepPoint[];
  memorySweepSeries: MemorySweepPoint[];
};
