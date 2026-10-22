import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Columns, SplitSquareVertical, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from 'lucide-react';

interface ImageComparisonSliderProps {
  originalUrl: string;
  enhancedUrl: string;
  width: number;
  height: number;
  labelLeft?: string;
  labelRight?: string;
}

export const ImageComparisonSlider: React.FC<ImageComparisonSliderProps> = ({
  originalUrl,
  enhancedUrl,
  width,
  height,
  labelLeft = 'Original (Input)',
  labelRight = 'Enhanced (Output)',
}) => {
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage 0 - 100
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [displayMode, setDisplayMode] = useState<'split' | 'side-by-side'>('split');
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const containerRef = useRef<HTMLDivElement>(null);

  const handlePointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    setIsDragging(true);
    updateSlider(e.clientX);
  };

  const updateSlider = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = Math.min(100, Math.max(0, (x / rect.width) * 100));
    setSliderPos(percent);
  }, []);

  useEffect(() => {
    const handlePointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      updateSlider(e.clientX);
    };

    const handlePointerUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      window.addEventListener('pointermove', handlePointerMove);
      window.addEventListener('pointerup', handlePointerUp);
    }
    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
    };
  }, [isDragging, updateSlider]);

  const handleZoom = (delta: number) => {
    setZoomLevel((prev) => Math.min(3, Math.max(0.75, Number((prev + delta).toFixed(2)))));
  };

  const handleResetZoom = () => {
    setZoomLevel(1);
  };

  return (
    <div className="flex flex-col gap-2 w-full">
      {/* Top Toolbar */}
      <div className="flex items-center justify-between px-2 py-1 bg-slate-900/60 rounded-xl border border-slate-800 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[11px] text-slate-400">
            {width} × {height} px ({(width * height).toLocaleString()} pixels)
          </span>
        </div>

        <div className="flex items-center gap-1.5 font-mono text-[11px]">
          {/* Split vs Side-by-Side Mode */}
          <button
            onClick={() => setDisplayMode('split')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-all ${
              displayMode === 'split' ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40' : 'hover:bg-slate-800 text-slate-400'
            }`}
            title="Split Slider Mode"
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            Split
          </button>
          <button
            onClick={() => setDisplayMode('side-by-side')}
            className={`flex items-center gap-1 px-2.5 py-1 rounded-md transition-all ${
              displayMode === 'side-by-side' ? 'bg-cyan-500/20 text-cyan-300 font-bold border border-cyan-500/40' : 'hover:bg-slate-800 text-slate-400'
            }`}
            title="Side-by-side View"
          >
            <Columns className="w-3.5 h-3.5" />
            Dual
          </button>

          {/* Zoom controls */}
          <div className="flex items-center gap-1 pl-2 ml-2 border-l border-slate-800">
            <button
              onClick={() => handleZoom(-0.25)}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <span className="text-[10px] w-9 text-center text-slate-300">
              {Math.round(zoomLevel * 100)}%
            </span>
            <button
              onClick={() => handleZoom(0.25)}
              className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-white"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            {zoomLevel !== 1 && (
              <button
                onClick={handleResetZoom}
                className="p-1 hover:bg-slate-800 rounded text-cyan-400"
                title="Reset Zoom"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Main Preview Container */}
      {displayMode === 'split' ? (
        <div
          ref={containerRef}
          onPointerDown={handlePointerDown}
          className="relative w-full aspect-[4/3] max-h-[520px] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800/80 shadow-2xl select-none cursor-ew-resize group"
        >
          {/* Background image: Enhanced */}
          <div
            className="absolute inset-0 flex items-center justify-center overflow-hidden"
            style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }}
          >
            <img
              src={enhancedUrl || originalUrl}
              alt={labelRight}
              className="max-w-full max-h-full object-contain pointer-events-none"
            />
          </div>

          {/* Foreground image: Original clipped by sliderPos */}
          <div
            className="absolute inset-0 overflow-hidden"
            style={{
              clipPath: `inset(0 ${100 - sliderPos}% 0 0)`,
            }}
          >
            <div
              className="w-full h-full flex items-center justify-center"
              style={{ transform: `scale(${zoomLevel})`, transformOrigin: 'center center' }}
            >
              <img
                src={originalUrl}
                alt={labelLeft}
                className="max-w-full max-h-full object-contain pointer-events-none"
              />
            </div>
          </div>

          {/* Draggable Divider Line & Handle */}
          <div
            className="absolute top-0 bottom-0 z-20 pointer-events-none"
            style={{ left: `${sliderPos}%` }}
          >
            <div className="absolute top-0 bottom-0 -ml-[1.5px] w-[3px] bg-gradient-to-b from-cyan-400 via-indigo-400 to-purple-400 shadow-[0_0_12px_rgba(56,189,248,0.7)]" />
            <div className="absolute top-1/2 -mt-4 -ml-4 w-8 h-8 rounded-full bg-slate-900 border-2 border-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/30">
              <div className="flex gap-0.5">
                <div className="w-0.5 h-3 bg-cyan-300 rounded-full" />
                <div className="w-0.5 h-3 bg-cyan-300 rounded-full" />
              </div>
            </div>
          </div>

          {/* Badges on left and right */}
          <div className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700/80 text-[11px] font-mono text-cyan-300 font-semibold shadow-md pointer-events-none">
            {labelLeft}
          </div>
          <div className="absolute top-3 right-3 z-10 px-2.5 py-1 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700/80 text-[11px] font-mono text-purple-300 font-semibold shadow-md pointer-events-none">
            {labelRight}
          </div>
        </div>
      ) : (
        /* Side by Side Mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full">
          <div className="relative aspect-[4/3] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center p-2">
            <div className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700 text-[11px] font-mono text-cyan-300 font-semibold">
              {labelLeft}
            </div>
            <img
              src={originalUrl}
              alt={labelLeft}
              className="max-w-full max-h-full object-contain"
              style={{ transform: `scale(${zoomLevel})` }}
            />
          </div>
          <div className="relative aspect-[4/3] bg-slate-950 rounded-2xl overflow-hidden border border-slate-800 flex items-center justify-center p-2">
            <div className="absolute top-3 left-3 z-10 px-2.5 py-1 rounded-lg bg-slate-900/80 backdrop-blur-md border border-slate-700 text-[11px] font-mono text-purple-300 font-semibold">
              {labelRight}
            </div>
            <img
              src={enhancedUrl || originalUrl}
              alt={labelRight}
              className="max-w-full max-h-full object-contain"
              style={{ transform: `scale(${zoomLevel})` }}
            />
          </div>
        </div>
      )}
    </div>
  );
};
