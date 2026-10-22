import React, { useState, useMemo } from 'react';
import { BarChart2, TrendingUp, Sliders } from 'lucide-react';

interface HistogramVisualizerProps {
  originalHist: number[];
  enhancedHist?: number[];
  width?: number;
  height?: number;
}

export const HistogramVisualizer: React.FC<HistogramVisualizerProps> = ({
  originalHist,
  enhancedHist,
}) => {
  const [hoverBin, setHoverBin] = useState<number | null>(null);
  const [viewMode, setViewMode] = useState<'both' | 'original' | 'enhanced'>('both');

  // Compute metrics
  const stats = useMemo(() => {
    const calcStats = (hist: number[]) => {
      let total = 0;
      let weightedSum = 0;
      let minVal = 255;
      let maxVal = 0;

      for (let i = 0; i < 256; i++) {
        const count = hist[i] || 0;
        total += count;
        weightedSum += i * count;
        if (count > 0) {
          if (i < minVal) minVal = i;
          if (i > maxVal) maxVal = i;
        }
      }

      const mean = total > 0 ? weightedSum / total : 0;
      let varianceSum = 0;
      for (let i = 0; i < 256; i++) {
        const count = hist[i] || 0;
        varianceSum += count * Math.pow(i - mean, 2);
      }
      const stdDev = total > 0 ? Math.sqrt(varianceSum / total) : 0;

      return {
        total,
        mean: mean.toFixed(1),
        stdDev: stdDev.toFixed(1),
        minVal: minVal === 255 ? 0 : minVal,
        maxVal,
        dynamicRange: maxVal - (minVal === 255 ? 0 : minVal),
      };
    };

    return {
      orig: calcStats(originalHist),
      enh: enhancedHist ? calcStats(enhancedHist) : null,
    };
  }, [originalHist, enhancedHist]);

  // Max peak for scaling
  const maxPeak = useMemo(() => {
    let m = 1;
    for (let i = 0; i < 256; i++) {
      if (originalHist[i] > m) m = originalHist[i];
      if (enhancedHist && enhancedHist[i] > m) m = enhancedHist[i];
    }
    return m;
  }, [originalHist, enhancedHist]);

  // SVG Coordinates
  const svgWidth = 600;
  const svgHeight = 150;
  const paddingBottom = 20;
  const usableHeight = svgHeight - paddingBottom;

  const getPoints = (hist: number[]) => {
    return hist
      .map((val, idx) => {
        const x = (idx / 255) * svgWidth;
        const y = usableHeight - (val / maxPeak) * (usableHeight - 8);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  };

  const origPolyline = useMemo(() => getPoints(originalHist), [originalHist, maxPeak]);
  const enhPolyline = useMemo(() => (enhancedHist ? getPoints(enhancedHist) : ''), [enhancedHist, maxPeak]);

  return (
    <div className="bg-slate-900/80 border border-slate-800/90 rounded-2xl p-4 shadow-xl backdrop-blur-md">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-cyan-400" />
          <h3 className="text-xs font-semibold text-slate-200 tracking-wide uppercase font-mono">
            256-Bin Intensity Histogram & Contrast CDF
          </h3>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex items-center bg-slate-950/80 p-0.5 rounded-lg border border-slate-800 text-[11px] font-mono">
            <button
              onClick={() => setViewMode('both')}
              className={`px-2 py-0.5 rounded ${viewMode === 'both' ? 'bg-cyan-500/20 text-cyan-300 font-bold' : 'text-slate-400'}`}
            >
              Overlay
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-2 py-0.5 rounded ${viewMode === 'original' ? 'bg-blue-500/20 text-blue-300 font-bold' : 'text-slate-400'}`}
            >
              Original
            </button>
            {enhancedHist && (
              <button
                onClick={() => setViewMode('enhanced')}
                className={`px-2 py-0.5 rounded ${viewMode === 'enhanced' ? 'bg-purple-500/20 text-purple-300 font-bold' : 'text-slate-400'}`}
              >
                Enhanced
              </button>
            )}
          </div>
        </div>
      </div>

      {/* SVG Canvas Chart */}
      <div className="relative w-full overflow-hidden bg-slate-950/90 rounded-xl border border-slate-800/80 p-2">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-36 cursor-crosshair"
          onMouseMove={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const relX = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
            const bin = Math.min(255, Math.floor((relX / rect.width) * 256));
            setHoverBin(bin);
          }}
          onMouseLeave={() => setHoverBin(null)}
        >
          <defs>
            <linearGradient id="origGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#38bdf8" stopOpacity="0.02" />
            </linearGradient>
            <linearGradient id="enhGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#c084fc" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#c084fc" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          <line x1="0" y1={usableHeight * 0.25} x2={svgWidth} y2={usableHeight * 0.25} stroke="#1e293b" strokeDasharray="3 3" />
          <line x1="0" y1={usableHeight * 0.5} x2={svgWidth} y2={usableHeight * 0.5} stroke="#1e293b" strokeDasharray="3 3" />
          <line x1="0" y1={usableHeight * 0.75} x2={svgWidth} y2={usableHeight * 0.75} stroke="#1e293b" strokeDasharray="3 3" />
          <line x1="0" y1={usableHeight} x2={svgWidth} y2={usableHeight} stroke="#334155" />

          {/* Original Histogram */}
          {(viewMode === 'both' || viewMode === 'original') && (
            <>
              <polygon
                points={`0,${usableHeight} ${origPolyline} ${svgWidth},${usableHeight}`}
                fill="url(#origGrad)"
              />
              <polyline
                points={origPolyline}
                fill="none"
                stroke="#38bdf8"
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
            </>
          )}

          {/* Enhanced Histogram */}
          {enhancedHist && (viewMode === 'both' || viewMode === 'enhanced') && (
            <>
              <polygon
                points={`0,${usableHeight} ${enhPolyline} ${svgWidth},${usableHeight}`}
                fill="url(#enhGrad)"
              />
              <polyline
                points={enhPolyline}
                fill="none"
                stroke="#c084fc"
                strokeWidth="1.8"
                strokeLinejoin="round"
              />
            </>
          )}

          {/* Hover indicator line */}
          {hoverBin !== null && (
            <line
              x1={(hoverBin / 255) * svgWidth}
              y1="0"
              x2={(hoverBin / 255) * svgWidth}
              y2={usableHeight}
              stroke="#f8fafc"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          )}

          {/* X Axis ticks */}
          <text x="5" y={svgHeight - 4} fill="#64748b" fontSize="10" fontFamily="monospace">0 (Black)</text>
          <text x={svgWidth / 2 - 20} y={svgHeight - 4} fill="#64748b" fontSize="10" fontFamily="monospace">128 (Mid)</text>
          <text x={svgWidth - 65} y={svgHeight - 4} fill="#64748b" fontSize="10" fontFamily="monospace">255 (White)</text>
        </svg>

        {/* Hover Readout Tooltip */}
        {hoverBin !== null && (
          <div className="absolute top-2 right-2 bg-slate-900/95 border border-cyan-500/40 rounded-lg px-2.5 py-1 text-[11px] font-mono shadow-lg flex items-center gap-3">
            <span className="text-slate-300">Bin: <strong className="text-white">{hoverBin}</strong></span>
            <span className="text-cyan-400">Orig: {originalHist[hoverBin]?.toLocaleString() || 0}</span>
            {enhancedHist && (
              <span className="text-purple-400">Enh: {enhancedHist[hoverBin]?.toLocaleString() || 0}</span>
            )}
          </div>
        )}
      </div>

      {/* Key Stats Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 pt-3 border-t border-slate-800 text-[11px] font-mono">
        <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
          <span className="text-slate-500 block">Original Mean / Std</span>
          <span className="font-semibold text-cyan-400">
            {stats.orig.mean} <span className="text-slate-600">±</span> {stats.orig.stdDev}
          </span>
        </div>
        <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
          <span className="text-slate-500 block">Original Range</span>
          <span className="font-semibold text-slate-300">
            [{stats.orig.minVal} - {stats.orig.maxVal}] (Δ{stats.orig.dynamicRange})
          </span>
        </div>
        {stats.enh ? (
          <>
            <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
              <span className="text-slate-500 block">Enhanced Mean / Std</span>
              <span className="font-semibold text-purple-400">
                {stats.enh.mean} <span className="text-slate-600">±</span> {stats.enh.stdDev}
              </span>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800/80">
              <span className="text-slate-500 block">Contrast Expansion</span>
              <span className="font-semibold text-emerald-400">
                +{Math.max(0, stats.enh.dynamicRange - stats.orig.dynamicRange)} levels
              </span>
            </div>
          </>
        ) : (
          <div className="col-span-2 p-2 rounded-lg bg-slate-950/30 border border-slate-800/40 text-slate-500 flex items-center justify-center">
            Run an enhancement to view equalized distribution metrics
          </div>
        )}
      </div>
    </div>
  );
};
