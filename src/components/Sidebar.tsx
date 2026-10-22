import React from 'react';
import {
  LayoutDashboard,
  Sparkles,
  Layers,
  BarChart3,
  CheckCircle2,
  BookOpen,
  FileCode,
  ShieldCheck
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      icon: LayoutDashboard,
      badge: 'Overview',
    },
    {
      id: 'enhancement',
      label: 'Image Enhancement',
      icon: Sparkles,
      badge: 'Interactive',
    },
    {
      id: 'architecture',
      label: 'CUDA Architecture',
      icon: Layers,
      badge: '20x20 Tile',
    },
    {
      id: 'benchmark',
      label: 'Performance Benchmark',
      icon: BarChart3,
      badge: '4K Multi-res',
    },
    {
      id: 'correctness',
      label: 'Correctness Tests',
      icon: CheckCircle2,
      badge: 'Pixel Check',
    },
    {
      id: 'docs',
      label: 'About Project & Viva',
      icon: BookOpen,
      badge: 'Full Report',
    },
  ];

  return (
    <aside className="w-full lg:w-64 shrink-0 p-4 border-b lg:border-b-0 lg:border-r border-slate-800/80 bg-[#090d16]/70">
      <div className="sticky top-20 flex flex-row lg:flex-col gap-1.5 overflow-x-auto pb-2 lg:pb-0 scrollbar-none">
        <div className="hidden lg:block px-3 py-2 text-[11px] font-mono uppercase tracking-wider text-slate-500 font-semibold">
          System Modules
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center justify-between gap-3 px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all whitespace-nowrap ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-500/15 via-indigo-500/15 to-purple-500/15 text-cyan-300 border border-cyan-500/30 shadow-sm shadow-cyan-500/10'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50 border border-transparent'
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </div>
              <span
                className={`hidden xl:inline text-[9px] font-mono px-1.5 py-0.5 rounded-md ${
                  isActive
                    ? 'bg-cyan-500/20 text-cyan-300 font-bold'
                    : 'bg-slate-800/80 text-slate-500'
                }`}
              >
                {item.badge}
              </span>
            </button>
          );
        })}

        <div className="hidden lg:block mt-6 pt-4 border-t border-slate-800/80 px-3">
          <div className="text-[11px] font-mono text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400" />
            CUDA Specifications
          </div>
          <div className="space-y-1.5 text-[11px] font-mono text-slate-400">
            <div className="flex justify-between py-1 border-b border-slate-900">
              <span className="text-slate-500">Block Dim:</span>
              <span className="text-slate-300">16 × 16 (256)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-900">
              <span className="text-slate-500">Shared Tile:</span>
              <span className="text-cyan-300">20 × 20 (400 elem)</span>
            </div>
            <div className="flex justify-between py-1 border-b border-slate-900">
              <span className="text-slate-500">Halo Radius:</span>
              <span className="text-slate-300">2 pixels (apron)</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-slate-500">Histogram:</span>
              <span className="text-indigo-300">256 Bins / Atomic</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
