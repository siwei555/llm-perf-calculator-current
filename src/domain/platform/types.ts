export type PlatformInput = {
  computeThroughputTflops: number;
  memoryBandwidthGbps: number;
  memoryCapacityGb: number;
  computeEfficiency: number;
  bandwidthEfficiency: number;
  /** Prefill 阶段 cache 流量相对于持久 cache 容量的估算比例，范围 0–1 */
  prefillCacheTrafficFactor: number;
  batchSize: number;
  /** 芯片数量；容量线性汇总，算力与带宽还需乘阶段相关的并行效率 */
  chipCount: number;
  prefillComputeParallelEfficiency: number;
  prefillBandwidthParallelEfficiency: number;
  decodeComputeParallelEfficiency: number;
  decodeBandwidthParallelEfficiency: number;
  /** 运行时额外显存的估算假设（GB），允许用户按实际运行框架手动调整 */
  runtimeOverheadGb: number;
  /** 每个权重参数的字节数（1 = FP8, 2 = BF16） */
  bytesPerWeight: number;
  /** 每个激活/cache 元素的字节数（2 = BF16） */
  bytesPerActivation: number;
  /** 每个专家权重的字节数（0.5 = FP4, 1 = FP8） */
  bytesPerExpert: number;
};
