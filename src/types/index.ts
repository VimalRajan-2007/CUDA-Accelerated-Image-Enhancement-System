/**
 * Core type definitions for CUDA Image Lab
 */

export type AlgorithmType = 
  | 'full_pipeline'
  | 'gaussian'
  | 'hist_eq'
  | 'sharpen'
  | 'gamma'
  | 'sobel';

export type TargetDevice = 'cuda' | 'cpu';

export interface TimingBreakdown {
  h2d_ms: number;
  kernel_ms: number;
  d2h_ms: number;
  total_ms: number;
  redundant_reads_avoided_pct?: number;
}

export interface ImageHistogram {
  original: number[];
  enhanced: number[];
}

export interface EnhancementResult {
  outputDataUrl: string;
  originalDataUrl: string;
  width: number;
  height: number;
  pixels: number;
  algorithm: AlgorithmType;
  targetDevice: TargetDevice;
  useSharedMemory: boolean;
  timings: TimingBreakdown;
  histogram: ImageHistogram;
  edgeMapDataUrl?: string;
  parameters: {
    sigma: number;
    amount: number;
    gamma: number;
  };
}

export interface BenchmarkRecord {
  resolution: string;
  pixels: number;
  cpu_gauss_ms: number;
  cpu_pipeline_ms: number;
  cuda_global_gauss_ms: number;
  cuda_shared_gauss_ms: number;
  cuda_pipeline_ms: number;
  cuda_kernel_ms: number;
  cuda_h2d_ms: number;
  cuda_d2h_ms: number;
  gauss_speedup: number;
  pipeline_speedup: number;
  device: string;
}

export interface VerificationResult {
  testCase: string;
  resolution: string;
  operation: string;
  tolerance: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  rmse: number;
  exactMatchPct: number;
  status: 'PASS' | 'FAIL';
  executionMode: string;
}

export interface HardwareStatus {
  cudaAvailable: boolean;
  executionMode: 'cuda_gpu' | 'cpu_fallback';
  gpuName: string;
  computeCapability: string;
  totalMemoryMB: number;
  initMessage: string;
}
