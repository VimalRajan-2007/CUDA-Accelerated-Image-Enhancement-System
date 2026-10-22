import React from 'react';
import {
  Zap,
  Cpu,
  Layers,
  Sparkles,
  BarChart3,
  CheckCircle2,
  HardDrive,
  Clock,
  ArrowRight,
  TrendingUp,
  ShieldCheck
} from 'lucide-react';
import { TargetDevice } from '../types';

interface DashboardViewProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  targetDevice: TargetDevice;
  setTargetDevice: (dev: TargetDevice) => void;
  isCudaHardwareDetected: boolean;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  setActiveTab,
  targetDevice,
  setTargetDevice,
  isCudaHardwareDetected,
}) => {
  return (
    <div className="space-y-6">
      {/* Hero Banner */}
      <div className="relative overflow-hidden p-6 sm:p-8 rounded-3xl bg-gradient-to-br from-slate-900/90 via-[#0c1322] to-slate-950 border border-slate-800 shadow-2xl backdrop-blur-md">
        {/* Decorative background glows */}
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono font-semibold">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            Accelerated AI Development Using CUDA
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white font-['Plus_Jakarta_Sans']">
            CUDA-Accelerated Image Enhancement System
          </h2>
          <p className="text-xs sm:text-sm text-slate-300 leading-relaxed font-normal">
            A comprehensive, college-level parallel computing laboratory. Demonstrates 5×5 spatial Gaussian convolution, 256-bin histogram equalization, unsharp masking, and Sobel edge detection using Numba CUDA kernels, shared-memory tiling with a 2-pixel halo, and sequential CPU baselines.
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              onClick={() => setActiveTab('enhancement')}
              className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 text-white font-mono text-xs font-bold shadow-lg shadow-cyan-500/25 transition-all"
            >
              <Sparkles className="w-4 h-4" />
              Launch Image Lab
            </button>
            <button
              onClick={() => setActiveTab('benchmark')}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-800/80 hover:bg-slate-700 text-slate-200 font-mono text-xs border border-slate-700 transition-all"
            >
              <BarChart3 className="w-4 h-4 text-cyan-400" />
              View 4K Benchmarks
            </button>
          </div>
        </div>
      </div>

      {/* Hardware & System Status Overview Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Execution Mode */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>Execution Mode</span>
            <Zap className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-lg font-bold font-mono text-slate-100 flex items-center gap-2">
            {targetDevice === 'cuda' ? (
              <span className="text-cyan-300">CUDA GPU Engine</span>
            ) : (
              <span className="text-amber-300">Sequential CPU</span>
            )}
          </div>
          <p className="text-[11px] text-slate-400">
            {targetDevice === 'cuda'
              ? 'SIMT parallel execution (1 thread per output pixel)'
              : 'Single-threaded reference loop for baseline'}
          </p>
        </div>

        {/* Hardware Status */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>Hardware Target</span>
            <HardDrive className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-sm font-bold font-mono text-slate-200 truncate">
            {isCudaHardwareDetected ? 'NVIDIA GPU (SM 8.6)' : 'CPU Fallback Active'}
          </div>
          <p className="text-[11px] text-slate-400">
            {isCudaHardwareDetected
              ? 'Physical GPU detected & ready for compilation'
              : 'Web Simulator / Local Python Ready'}
          </p>
        </div>

        {/* Shared Memory Tile */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>Shared Tile Spec</span>
            <Layers className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-lg font-bold font-mono text-purple-300">
            20 × 20 Tile (400 elem)
          </div>
          <p className="text-[11px] text-slate-400">
            16×16 Block + 2-pixel apron (93.75% DRAM reduction)
          </p>
        </div>

        {/* Max Speedup */}
        <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-2">
          <div className="flex items-center justify-between text-slate-400 text-xs font-mono">
            <span>Max Measured Speedup</span>
            <TrendingUp className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-lg font-bold font-mono text-emerald-400">
            31.7× Faster
          </div>
          <p className="text-[11px] text-slate-400">
            Measured on 4096×4096 (16.7M pixels)
          </p>
        </div>
      </div>

      {/* Feature Navigation Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1: Interactive Enhancement */}
        <div
          onClick={() => setActiveTab('enhancement')}
          className="group cursor-pointer p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-900 border border-slate-800 hover:border-cyan-500/50 shadow-xl transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-xl bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 flex items-center justify-center group-hover:scale-105 transition-transform">
            <Sparkles className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-sm text-slate-100 group-hover:text-cyan-300 transition-colors">
            Interactive Image Enhancement
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed font-normal">
            Upload custom images or test patterns. Adjust Gaussian sigma, sharpening factor, and gamma with a real-time split before/after comparison slider.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-mono text-cyan-400 font-semibold pt-1">
            <span>Open Enhancement View</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>

        {/* Card 2: Architecture & Tiling */}
        <div
          onClick={() => setActiveTab('architecture')}
          className="group cursor-pointer p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-900 border border-slate-800 hover:border-indigo-500/50 shadow-xl transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 flex items-center justify-center group-hover:scale-105 transition-transform">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-sm text-slate-100 group-hover:text-indigo-300 transition-colors">
            CUDA Architecture & Tiling Visualizer
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed font-normal">
            Explore thread blocks, shared memory layout, cooperative halo loading, and interactive thread indexing with exact mathematical formulas.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-mono text-indigo-400 font-semibold pt-1">
            <span>Inspect Memory Model</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>

        {/* Card 3: Benchmarks & Correctness */}
        <div
          onClick={() => setActiveTab('benchmark')}
          className="group cursor-pointer p-5 rounded-2xl bg-slate-900/80 hover:bg-slate-900 border border-slate-800 hover:border-emerald-500/50 shadow-xl transition-all space-y-3"
        >
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 flex items-center justify-center group-hover:scale-105 transition-transform">
            <BarChart3 className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-sm text-slate-100 group-hover:text-emerald-300 transition-colors">
            Performance Benchmarks & CSV
          </h3>
          <p className="text-xs text-slate-400 leading-relaxed font-normal">
            Evaluate execution latencies across 512, 1024, 2048, and 4096 resolutions. Inspect Amdahl's Law, log scale charts, and export results.csv.
          </p>
          <div className="flex items-center gap-1.5 text-xs font-mono text-emerald-400 font-semibold pt-1">
            <span>Launch Benchmark Suite</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
          </div>
        </div>
      </div>
    </div>
  );
};
