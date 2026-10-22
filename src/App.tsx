import React, { useState, useEffect } from 'react';
import { TargetDevice } from './types';
import { Navbar } from './components/Navbar';
import { Sidebar } from './components/Sidebar';
import { DashboardView } from './components/DashboardView';
import { ImageEnhancementView } from './components/ImageEnhancementView';
import { CudaArchitectureView } from './components/CudaArchitectureView';
import { BenchmarkView } from './components/BenchmarkView';
import { CorrectnessView } from './components/CorrectnessView';
import { DocumentationView } from './components/DocumentationView';

export default function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [targetDevice, setTargetDevice] = useState<TargetDevice>('cuda');
  const [isCudaHardwareDetected, setIsCudaHardwareDetected] = useState<boolean>(false);

  // Probe server status if available
  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => {
        if (data && data.cuda_available) {
          setIsCudaHardwareDetected(true);
        } else {
          setIsCudaHardwareDetected(false);
        }
      })
      .catch(() => {
        // Standalone SPA client fallback
        setIsCudaHardwareDetected(false);
      });
  }, []);

  return (
    <div className="min-h-screen bg-[#070a12] text-slate-100 flex flex-col font-['Plus_Jakarta_Sans'] selection:bg-cyan-500 selection:text-black">
      {/* Top Navigation Bar */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        targetDevice={targetDevice}
        setTargetDevice={setTargetDevice}
        isCudaHardwareDetected={isCudaHardwareDetected}
      />

      {/* Main Body with Sidebar + Content */}
      <div className="flex-1 max-w-7xl w-full mx-auto flex flex-col lg:flex-row">
        {/* Navigation Sidebar */}
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        {/* Dynamic Content View */}
        <main className="flex-1 p-4 sm:p-6 lg:p-8 overflow-y-auto">
          {activeTab === 'dashboard' && (
            <DashboardView
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              targetDevice={targetDevice}
              setTargetDevice={setTargetDevice}
              isCudaHardwareDetected={isCudaHardwareDetected}
            />
          )}

          {activeTab === 'enhancement' && (
            <ImageEnhancementView
              targetDevice={targetDevice}
              setTargetDevice={setTargetDevice}
              isCudaHardwareDetected={isCudaHardwareDetected}
            />
          )}

          {activeTab === 'architecture' && <CudaArchitectureView />}

          {activeTab === 'benchmark' && (
            <BenchmarkView isCudaHardwareDetected={isCudaHardwareDetected} />
          )}

          {activeTab === 'correctness' && (
            <CorrectnessView isCudaHardwareDetected={isCudaHardwareDetected} />
          )}

          {activeTab === 'docs' && <DocumentationView />}
        </main>
      </div>

      {/* Footer */}
      <footer className="w-full border-t border-slate-900 bg-[#05070c] py-4 px-6 text-center text-xs font-mono text-slate-500">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>CUDA-Accelerated Image Enhancement System — B.Tech Artificial Intelligence & Data Science</span>
          <div className="flex items-center gap-4 text-slate-400">
            <button onClick={() => setActiveTab('docs')} className="hover:text-cyan-400 transition-colors">
              Project Report
            </button>
            <span>•</span>
            <button onClick={() => setActiveTab('architecture')} className="hover:text-cyan-400 transition-colors">
              Tiling Architecture
            </button>
            <span>•</span>
            <button onClick={() => setActiveTab('benchmark')} className="hover:text-cyan-400 transition-colors">
              Benchmark CSV
            </button>
          </div>
        </div>
      </footer>
    </div>
  );
}
