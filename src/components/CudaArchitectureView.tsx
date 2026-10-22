import React, { useState } from 'react';
import { Layers, Cpu, Zap, ArrowRight, Database, Server, Clock, ShieldCheck, CheckCircle2 } from 'lucide-react';

export const CudaArchitectureView: React.FC = () => {
  const [selectedTx, setSelectedTx] = useState<number>(3);
  const [selectedTy, setSelectedTy] = useState<number>(4);
  const [blockIdxX, setBlockIdxX] = useState<number>(2);
  const [blockIdxY, setBlockIdxY] = useState<number>(1);

  // Computed values
  const tid = selectedTy * 16 + selectedTx;
  const globalRow = blockIdxY * 16 + selectedTy;
  const globalCol = blockIdxX * 16 + selectedTx;
  const sharedCenterR = selectedTy + 2;
  const sharedCenterC = selectedTx + 2;

  // Gaussian weights 5x5
  const gaussianWeights5x5 = [
    [0.003, 0.013, 0.022, 0.013, 0.003],
    [0.013, 0.059, 0.097, 0.059, 0.013],
    [0.022, 0.097, 0.159, 0.097, 0.022],
    [0.013, 0.059, 0.097, 0.059, 0.013],
    [0.003, 0.013, 0.022, 0.013, 0.003],
  ];

  return (
    <div className="space-y-8">
      {/* Overview Card */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800/90 shadow-2xl backdrop-blur-md">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-9 h-9 rounded-xl bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 flex items-center justify-center">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-slate-100">
              CUDA Memory & Execution Hierarchy
            </h2>
            <p className="text-xs text-slate-400">
              Educational visualization of parallel thread blocks, shared-memory tiling, and cooperative loading.
            </p>
          </div>
        </div>

        {/* Pipeline Stage Flowchart */}
        <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2 text-center text-xs font-mono">
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-slate-500 block text-[10px] uppercase">Stage 1</span>
            <span className="font-bold text-slate-200 mt-1 block">Host RAM</span>
            <small className="text-slate-500 text-[10px]">CPU Memory</small>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-cyan-500/30 text-cyan-300">
            <span className="text-cyan-500 block text-[10px] uppercase">Stage 2 (PCIe)</span>
            <span className="font-bold text-cyan-300 mt-1 block">cudaMemcpy</span>
            <small className="text-slate-400 text-[10px]">Host-to-Device</small>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800">
            <span className="text-slate-500 block text-[10px] uppercase">Stage 3</span>
            <span className="font-bold text-slate-200 mt-1 block">Global VRAM</span>
            <small className="text-slate-500 text-[10px]">High Latency (400c)</small>
          </div>
          <div className="p-3 rounded-xl bg-indigo-950/40 border border-indigo-500/40 text-indigo-300">
            <span className="text-indigo-400 block text-[10px] uppercase">Stage 4 (Coop)</span>
            <span className="font-bold text-indigo-300 mt-1 block">Shared Tile</span>
            <small className="text-slate-400 text-[10px]">20×20 SRAM (1-5c)</small>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-purple-500/30 text-purple-300">
            <span className="text-purple-400 block text-[10px] uppercase">Stage 5</span>
            <span className="font-bold text-purple-300 mt-1 block">CUDA Cores</span>
            <small className="text-slate-400 text-[10px]">16×16 Threads</small>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-cyan-500/30 text-cyan-300">
            <span className="text-cyan-500 block text-[10px] uppercase">Stage 6 (PCIe)</span>
            <span className="font-bold text-cyan-300 mt-1 block">cudaMemcpy</span>
            <small className="text-slate-400 text-[10px]">Device-to-Host</small>
          </div>
          <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-500/40 text-emerald-300">
            <span className="text-emerald-400 block text-[10px] uppercase">Stage 7</span>
            <span className="font-bold text-emerald-300 mt-1 block">Enhanced Img</span>
            <small className="text-slate-400 text-[10px]">Ready for Display</small>
          </div>
        </div>
      </div>

      {/* Comparison: CPU Architecture vs GPU Architecture */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* CPU Card */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-slate-200">
            <Cpu className="w-5 h-5 text-amber-400" />
            <h3 className="font-bold text-sm uppercase tracking-wider font-mono">
              Central Processing Unit (CPU)
            </h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Optimized for low-latency serial execution of complex logic and branch-heavy code. Uses large L2/L3 caches and aggressive speculative execution units.
          </p>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono space-y-1.5 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">Core Count:</span>
              <span>4 to 16 Heavyweight Cores</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Execution Paradigm:</span>
              <span>SIMD (AVX-512 / NEON)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Memory Bandwidth:</span>
              <span>~50 - 90 GB/s (DDR5)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Image Filtering:</span>
              <span className="text-amber-400">Sequential loop over H × W pixels</span>
            </div>
          </div>
        </div>

        {/* GPU Card */}
        <div className="p-5 rounded-2xl bg-slate-900/80 border border-cyan-500/30 shadow-xl space-y-3">
          <div className="flex items-center gap-2.5 text-cyan-300">
            <Zap className="w-5 h-5 text-cyan-400" />
            <h3 className="font-bold text-sm uppercase tracking-wider font-mono">
              Graphics Processing Unit (GPU / CUDA)
            </h3>
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            Massively parallel throughput-oriented architecture designed to execute millions of lightweight threads concurrently using Streaming Multiprocessors (SMs).
          </p>
          <div className="p-3 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono space-y-1.5 text-slate-300">
            <div className="flex justify-between">
              <span className="text-slate-500">CUDA Cores:</span>
              <span className="text-cyan-300">1,500 to 16,384+ Cores</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Execution Paradigm:</span>
              <span>SIMT (Single Instruction, Multiple Threads)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Memory Bandwidth:</span>
              <span className="text-cyan-300">~360 - 1,000+ GB/s (GDDR6X)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Image Filtering:</span>
              <span className="text-emerald-400">1 Thread per Output Pixel</span>
            </div>
          </div>
        </div>
      </div>

      {/* Shared-Memory Tiling: 16x16 Output vs 20x20 Input Tile */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-2xl space-y-6">
        <div>
          <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            Shared-Memory Tiling & Halo Region (20×20 Input vs 16×16 Output)
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Why does Shared-Memory Tiling outperform naive Global Memory convolution by 16× in DRAM traffic?
          </p>
        </div>

        {/* The Mathematical Comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-4 rounded-xl bg-slate-950/90 border border-red-500/20 text-xs font-mono space-y-2">
            <span className="text-red-400 font-bold block uppercase">
              1. Naive Global Memory Approach
            </span>
            <p className="text-slate-400 text-[11px] font-sans">
              Every thread independently fetches 25 pixels from off-chip DRAM (5×5 stencil):
            </p>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-[11px]">
              DRAM Reads per block = 16 × 16 × 25 = <strong className="text-red-400">6,400 Reads</strong>
            </div>
            <p className="text-slate-500 text-[11px]">
              High contention on the memory bus; identical pixels are re-read up to 25 times by adjacent threads.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-950/90 border border-emerald-500/30 text-xs font-mono space-y-2">
            <span className="text-emerald-400 font-bold block uppercase">
              2. Shared-Memory Tiled Optimization
            </span>
            <p className="text-slate-400 text-[11px] font-sans">
              The 256 threads in the block cooperatively load a 20×20 tile (including 2-pixel halo) into on-chip SRAM:
            </p>
            <div className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 text-[11px]">
              DRAM Reads per block = 20 × 20 = <strong className="text-emerald-400">400 Reads (93.75% Reduction!)</strong>
            </div>
            <p className="text-emerald-300/80 text-[11px]">
              Once in shared memory, threads synchronize with <code>cuda.syncthreads()</code> and read at ~1.5 TB/s SRAM speeds.
            </p>
          </div>
        </div>

        {/* Cooperative Loading Two-Pass Breakdown */}
        <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800 text-xs font-mono space-y-3">
          <span className="text-indigo-300 font-bold block uppercase">
            Cooperative Loading Algorithm (256 Threads Loading 400 Shared Tile Pixels)
          </span>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px]">
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
              <span className="text-cyan-400 font-semibold block mb-1">Pass 1: Threads 0 to 255</span>
              <p className="text-slate-400 font-sans">
                Each of the 256 threads loads element <code>tid</code> (where <code>tile_r = tid / 20</code> and <code>tile_c = tid % 20</code>).
              </p>
            </div>
            <div className="p-3 rounded-lg bg-slate-900/80 border border-slate-800">
              <span className="text-purple-400 font-semibold block mb-1">Pass 2: First 144 Threads (0 to 143)</span>
              <p className="text-slate-400 font-sans">
                Loads the remaining <code>tid + 256</code> elements (from 256 to 399) to fully populate the 20×20 shared tile.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Thread Indexing & Worked Example */}
      <div className="p-6 rounded-2xl bg-slate-900/80 border border-slate-800 shadow-2xl space-y-6">
        <div>
          <h3 className="text-sm font-bold text-slate-100 uppercase tracking-wider font-mono flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-cyan-400" />
            Interactive Thread & Stencil Arithmetic Explorer
          </h3>
          <p className="text-xs text-slate-400 mt-1">
            Pick a thread inside a 16×16 block to trace its exact 2D coordinates, shared memory offset, and 5×5 convolution weights.
          </p>
        </div>

        {/* Sliders for (tx, ty) and (bx, by) */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-950/80 border border-slate-800 text-xs font-mono">
          <div>
            <span className="text-slate-400 block mb-1">threadIdx.x: <strong className="text-cyan-400">{selectedTx}</strong></span>
            <input
              type="range"
              min="0"
              max="15"
              value={selectedTx}
              onChange={(e) => setSelectedTx(parseInt(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>
          <div>
            <span className="text-slate-400 block mb-1">threadIdx.y: <strong className="text-cyan-400">{selectedTy}</strong></span>
            <input
              type="range"
              min="0"
              max="15"
              value={selectedTy}
              onChange={(e) => setSelectedTy(parseInt(e.target.value))}
              className="w-full accent-cyan-500 cursor-pointer"
            />
          </div>
          <div>
            <span className="text-slate-400 block mb-1">blockIdx.x: <strong className="text-purple-400">{blockIdxX}</strong></span>
            <input
              type="range"
              min="0"
              max="15"
              value={blockIdxX}
              onChange={(e) => setBlockIdxX(parseInt(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
          </div>
          <div>
            <span className="text-slate-400 block mb-1">blockIdx.y: <strong className="text-purple-400">{blockIdxY}</strong></span>
            <input
              type="range"
              min="0"
              max="15"
              value={blockIdxY}
              onChange={(e) => setBlockIdxY(parseInt(e.target.value))}
              className="w-full accent-purple-500 cursor-pointer"
            />
          </div>
        </div>

        {/* Live Coordinate Equations */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs font-mono">
          <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800">
            <span className="text-slate-500 block text-[10px]">Linear Thread ID</span>
            <span className="text-slate-200 mt-1 block">
              tid = ty × 16 + tx = {selectedTy} × 16 + {selectedTx} = <strong className="text-cyan-300">{tid}</strong>
            </span>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800">
            <span className="text-slate-500 block text-[10px]">Global Image Pixel (r, c)</span>
            <span className="text-slate-200 mt-1 block">
              row = {blockIdxY}×16 + {selectedTy} = <strong className="text-purple-300">{globalRow}</strong>, col = {blockIdxX}×16 + {selectedTx} = <strong className="text-purple-300">{globalCol}</strong>
            </span>
          </div>
          <div className="p-3 rounded-xl bg-slate-950/90 border border-slate-800">
            <span className="text-slate-500 block text-[10px]">Shared Tile Center</span>
            <span className="text-slate-200 mt-1 block">
              s_tile[{selectedTy} + 2, {selectedTx} + 2] = <strong className="text-emerald-300">s_tile[{sharedCenterR}, {sharedCenterC}]</strong>
            </span>
          </div>
        </div>

        {/* 5x5 Kernel Matrix Display */}
        <div className="p-4 rounded-xl bg-slate-950/90 border border-slate-800">
          <span className="text-xs font-mono font-bold text-slate-300 block mb-2">
            5×5 Normalized Gaussian Convolution Kernel (σ = 1.0)
          </span>
          <div className="overflow-x-auto">
            <table className="table-auto text-center font-mono text-[11px] w-full max-w-lg mx-auto border-collapse">
              <tbody>
                {gaussianWeights5x5.map((row, r) => (
                  <tr key={r}>
                    {row.map((val, c) => (
                      <td
                        key={c}
                        className={`p-2 border border-slate-800/80 ${
                          r === 2 && c === 2
                            ? 'bg-cyan-500/20 text-cyan-300 font-bold border-cyan-500/40'
                            : 'bg-slate-900/50 text-slate-400'
                        }`}
                      >
                        {val.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-[11px] text-slate-500 text-center mt-2 font-mono">
            Output Pixel = Σ (s_tile[r + kr, c + kc] × Weight[kr+2, kc+2]) for kr, kc ∈ [-2, 2]
          </p>
        </div>
      </div>
    </div>
  );
};
