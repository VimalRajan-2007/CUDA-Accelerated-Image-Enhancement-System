import React from 'react';
import { Cpu, Zap, Activity, HardDrive, Terminal } from 'lucide-react';
import { TargetDevice } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  targetDevice: TargetDevice;
  setTargetDevice: (dev: TargetDevice) => void;
  isCudaHardwareDetected: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  targetDevice,
  setTargetDevice,
  isCudaHardwareDetected,
}) => {
  return (
    <header className="sticky top-0 z-40 w-full border-b border-slate-800 bg-[#090d16]/90 backdrop-blur-md px-4 sm:px-6 py-3">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-cyan-400/30">
            <Zap className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent font-['Plus_Jakarta_Sans']">
                CUDA IMAGE LAB
              </h1>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-950/80 text-cyan-300 border border-cyan-800/60 uppercase tracking-wider font-semibold">
                B.Tech AI&DS
              </span>
            </div>
            <p className="text-xs text-slate-400 tracking-wide font-normal">
              GPU-Accelerated Image Enhancement & Performance Analysis System
            </p>
          </div>
        </div>

        {/* Execution Mode & Hardware Badge */}
        <div className="flex items-center flex-wrap gap-2.5">
          {/* Hardware Status Tag */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300">
            <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-mono text-slate-400">Target Host:</span>
            {isCudaHardwareDetected ? (
              <span className="flex items-center gap-1.5 text-emerald-400 font-semibold font-mono">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                NVIDIA GPU (SM 8.6 / 16x16 Warps)
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-amber-300 font-medium font-mono">
                <span className="w-2 h-2 rounded-full bg-amber-400"></span>
                CPU Mode (CUDA Sim / Python Ready)
              </span>
            )}
          </div>

          {/* Engine Selector Buttons */}
          <div className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800 shadow-inner">
            <button
              onClick={() => setTargetDevice('cuda')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                targetDevice === 'cuda'
                  ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-md shadow-cyan-600/30'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              CUDA GPU
            </button>
            <button
              onClick={() => setTargetDevice('cpu')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                targetDevice === 'cpu'
                  ? 'bg-slate-700 text-white shadow-md'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Cpu className="w-3.5 h-3.5" />
              Sequential CPU
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
