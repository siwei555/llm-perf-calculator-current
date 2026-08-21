import type { PlatformInput } from "../../../domain/platform/types";

export type ScalingPhase = "prefill" | "decode";
export type ScalingResource = "compute" | "bandwidth";

export const parallelEfficiencyDefaults: Record<1 | 2 | 4, Pick<PlatformInput,
  "prefillComputeParallelEfficiency" |
  "prefillBandwidthParallelEfficiency" |
  "decodeComputeParallelEfficiency" |
  "decodeBandwidthParallelEfficiency"
>> = {
  1: { prefillComputeParallelEfficiency: 1, prefillBandwidthParallelEfficiency: 1, decodeComputeParallelEfficiency: 1, decodeBandwidthParallelEfficiency: 1 },
  2: { prefillComputeParallelEfficiency: 0.72, prefillBandwidthParallelEfficiency: 0.78, decodeComputeParallelEfficiency: 0.65, decodeBandwidthParallelEfficiency: 0.70 },
  4: { prefillComputeParallelEfficiency: 0.58, prefillBandwidthParallelEfficiency: 0.64, decodeComputeParallelEfficiency: 0.48, decodeBandwidthParallelEfficiency: 0.52 }
};

export function defaultParallelEfficiencies(chipCount: number) {
  if (chipCount <= 1) return parallelEfficiencyDefaults[1];
  if (chipCount === 2) return parallelEfficiencyDefaults[2];
  if (chipCount === 4) return parallelEfficiencyDefaults[4];
  return {
    prefillComputeParallelEfficiency: Math.max(0.35, 0.72 * chipCount ** -0.18),
    prefillBandwidthParallelEfficiency: Math.max(0.40, 0.78 * chipCount ** -0.15),
    decodeComputeParallelEfficiency: Math.max(0.28, 0.65 * chipCount ** -0.22),
    decodeBandwidthParallelEfficiency: Math.max(0.32, 0.70 * chipCount ** -0.20)
  };
}

export function getParallelEfficiency(platform: PlatformInput, phase: ScalingPhase, resource: ScalingResource) {
  if (platform.chipCount <= 1) return 1;
  const key = `${phase}${resource === "compute" ? "Compute" : "Bandwidth"}ParallelEfficiency` as const;
  const configured = platform[key];
  return Number.isFinite(configured) ? configured : defaultParallelEfficiencies(platform.chipCount)[key];
}

export function getResourceScaling(platform: PlatformInput, phase: ScalingPhase, resource: ScalingResource) {
  return Math.max(1, platform.chipCount) * getParallelEfficiency(platform, phase, resource);
}

export function getTotalMemoryCapacityGb(platform: PlatformInput) {
  return platform.memoryCapacityGb * Math.max(1, platform.chipCount);
}
