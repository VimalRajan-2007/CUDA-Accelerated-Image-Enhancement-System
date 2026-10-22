import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Upload,
  Image as ImageIcon,
  Play,
  RotateCcw,
  Download,
  Zap,
  Cpu,
  Layers,
  Sliders,
  CheckCircle,
  Clock,
  ArrowRight,
  ShieldAlert,
  FileCheck2
} from 'lucide-react';
import { AlgorithmType, TargetDevice, EnhancementResult } from '../types';
import {
  rgbaToGrayscale,
  grayscaleToImageData,
  imageDataToDataUrl,
  executeEnhancement,
  applySobel,
} from '../utils/cudaSim';
import { SAMPLE_IMAGES } from '../utils/sampleImages';
import { ImageComparisonSlider } from './ImageComparisonSlider';
import { HistogramVisualizer } from './HistogramVisualizer';

interface ImageEnhancementViewProps {
  targetDevice: TargetDevice;
  setTargetDevice: (dev: TargetDevice) => void;
  isCudaHardwareDetected: boolean;
  onRunComplete?: (result: EnhancementResult) => void;
}

export const ImageEnhancementView: React.FC<ImageEnhancementViewProps> = ({
  targetDevice,
  setTargetDevice,
  isCudaHardwareDetected,
  onRunComplete,
}) => {
  // Image State
  const [currentGray, setCurrentGray] = useState<Uint8Array | null>(null);
  const [imageWidth, setImageWidth] = useState<number>(512);
  const [imageHeight, setImageHeight] = useState<number>(512);
  const [originalDataUrl, setOriginalDataUrl] = useState<string>('');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('brain_mri');

  // Controls State
  const [algorithm, setAlgorithm] = useState<AlgorithmType>('full_pipeline');
  const [useSharedMemory, setUseSharedMemory] = useState<boolean>(true);
  const [sigma, setSigma] = useState<number>(1.0);
  const [amount, setAmount] = useState<number>(1.5);
  const [gamma, setGamma] = useState<number>(0.6);

  // Result & Execution State
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [result, setResult] = useState<EnhancementResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load sample image on mount or change
  useEffect(() => {
    loadSampleImage(selectedSampleId);
  }, [selectedSampleId]);

  const loadSampleImage = (id: string) => {
    const sample = SAMPLE_IMAGES.find((s) => s.id === id);
    if (!sample) return;

    const imgData = sample.generate();
    const gray = rgbaToGrayscale(imgData.data, sample.width, sample.height);
    const dataUrl = imageDataToDataUrl(imgData);

    setCurrentGray(gray);
    setImageWidth(sample.width);
    setImageHeight(sample.height);
    setOriginalDataUrl(dataUrl);
    setResult(null);
    setErrorMessage('');
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match(/^image\/(png|jpeg|jpg|tiff|webp)$/)) {
      setErrorMessage('Unsupported file format. Please upload PNG, JPG, or TIFF image.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        // Limit max upload dimension for client responsiveness
        let w = img.width;
        let h = img.height;
        const maxDim = 1024;
        if (w > maxDim || h > maxDim) {
          if (w > h) {
            h = Math.round((h * maxDim) / w);
            w = maxDim;
          } else {
            w = Math.round((w * maxDim) / h);
            h = maxDim;
          }
        }

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(img, 0, 0, w, h);
        const imgData = ctx.getImageData(0, 0, w, h);

        const gray = rgbaToGrayscale(imgData.data, w, h);
        const grayImgData = grayscaleToImageData(gray, w, h);
        const dataUrl = imageDataToDataUrl(grayImgData);

        setCurrentGray(gray);
        setImageWidth(w);
        setImageHeight(h);
        setOriginalDataUrl(dataUrl);
        setResult(null);
        setErrorMessage('');
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  // Run enhancement operation
  const runEnhancement = (forcedAlgo?: AlgorithmType) => {
    if (!currentGray) return;
    const activeAlgo = forcedAlgo || algorithm;

    setIsProcessing(true);
    setErrorMessage('');

    setTimeout(() => {
      try {
        const res = executeEnhancement(
          currentGray,
          imageWidth,
          imageHeight,
          activeAlgo,
          targetDevice,
          useSharedMemory,
          { sigma, amount, gamma }
        );

        const enhancedImgData = grayscaleToImageData(res.enhanced, imageWidth, imageHeight);
        const enhancedUrl = imageDataToDataUrl(enhancedImgData);

        let edgeUrl: string | undefined;
        if (res.edgeMap) {
          const edgeImgData = grayscaleToImageData(res.edgeMap, imageWidth, imageHeight);
          edgeUrl = imageDataToDataUrl(edgeImgData);
        }

        const enhResult: EnhancementResult = {
          outputDataUrl: enhancedUrl,
          originalDataUrl,
          width: imageWidth,
          height: imageHeight,
          pixels: imageWidth * imageHeight,
          algorithm: activeAlgo,
          targetDevice,
          useSharedMemory,
          timings: res.timings,
          histogram: res.histogram,
          edgeMapDataUrl: edgeUrl,
          parameters: { sigma, amount, gamma },
        };

        setResult(enhResult);
        onRunComplete?.(enhResult);
      } catch (err: any) {
        setErrorMessage(`Processing error: ${err.message}`);
      } finally {
        setIsProcessing(false);
      }
    }, 40);
  };

  const handleDownload = (dataUrl: string, filename: string) => {
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="space-y-6">
      {/* Header and Sample Selection */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-4 rounded-2xl bg-slate-900/60 border border-slate-800/80 backdrop-blur-md">
        <div>
          <h2 className="text-lg font-bold text-slate-100 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-cyan-400" />
            Interactive CUDA Enhancement Pipeline
          </h2>
          <p className="text-xs text-slate-400">
            Upload custom imagery or select scientific test patterns to benchmark parallel GPU filtering versus sequential CPU execution.
          </p>
        </div>

        {/* Sample Gallery Buttons */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 lg:pb-0 scrollbar-none">
          <span className="text-xs font-mono text-slate-500 whitespace-nowrap">Sample:</span>
          {SAMPLE_IMAGES.map((sample) => (
            <button
              key={sample.id}
              onClick={() => setSelectedSampleId(sample.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-mono transition-all whitespace-nowrap ${
                selectedSampleId === sample.id
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/50 shadow-sm font-semibold'
                  : 'bg-slate-950/70 text-slate-400 border border-slate-800 hover:text-slate-200'
              }`}
            >
              {sample.name}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid: Controls on left, Visualizer on right */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Controls Sidebar (4 cols) */}
        <div className="lg:col-span-4 space-y-4">
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800/90 shadow-xl backdrop-blur-md space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Algorithm & Hardware
              </span>
              <button
                onClick={() => {
                  setSigma(1.0);
                  setAmount(1.5);
                  setGamma(0.6);
                  setUseSharedMemory(true);
                }}
                className="text-[11px] font-mono text-slate-400 hover:text-cyan-400 flex items-center gap-1"
                title="Reset sliders to defaults"
              >
                <RotateCcw className="w-3 h-3" />
                Reset
              </button>
            </div>

            {/* Algorithm Select */}
            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1.5 font-semibold">
                Enhancement Algorithm
              </label>
              <select
                value={algorithm}
                onChange={(e) => setAlgorithm(e.target.value as AlgorithmType)}
                className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono shadow-inner"
              >
                <option value="full_pipeline">Full 7-Step Pipeline (Recommended)</option>
                <option value="gaussian">5×5 Gaussian Denoising</option>
                <option value="hist_eq">Histogram Equalization (CDF Remapping)</option>
                <option value="sharpen">Unsharp-Mask Sharpening</option>
                <option value="gamma">Gamma Contrast Correction</option>
                <option value="sobel">Sobel Gradient Edge Detection</option>
              </select>
            </div>

            {/* Target Hardware Selector */}
            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1.5 font-semibold">
                Execution Architecture
              </label>
              <div className="grid grid-cols-2 gap-2">
                <button
                  onClick={() => setTargetDevice('cuda')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-mono font-bold transition-all ${
                    targetDevice === 'cuda'
                      ? 'bg-gradient-to-r from-cyan-600 to-indigo-600 text-white shadow-md shadow-cyan-600/25 border border-cyan-400/40'
                      : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'
                  }`}
                >
                  <Zap className="w-3.5 h-3.5" />
                  CUDA GPU
                </button>
                <button
                  onClick={() => setTargetDevice('cpu')}
                  className={`flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-xs font-mono font-bold transition-all ${
                    targetDevice === 'cpu'
                      ? 'bg-slate-700 text-white shadow-md border border-slate-600'
                      : 'bg-slate-950 text-slate-400 border border-slate-800 hover:text-slate-200'
                  }`}
                >
                  <Cpu className="w-3.5 h-3.5" />
                  CPU Single-Thread
                </button>
              </div>
            </div>

            {/* Shared Memory Switch (Only for CUDA) */}
            {targetDevice === 'cuda' && (
              <div className="p-3 rounded-xl bg-slate-950/70 border border-indigo-500/20 space-y-1.5">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    <span className="text-xs font-mono font-semibold text-slate-200">
                      Shared-Memory Tiling
                    </span>
                  </div>
                  <input
                    type="checkbox"
                    id="sharedToggle"
                    checked={useSharedMemory}
                    onChange={(e) => setUseSharedMemory(e.target.checked)}
                    className="w-4 h-4 accent-cyan-500 rounded cursor-pointer"
                  />
                </div>
                <p className="text-[11px] text-slate-400 font-sans">
                  {useSharedMemory
                    ? '20×20 shared tile with 2-pixel halo. Reduces DRAM traffic by ~93.7% via cooperative loading.'
                    : 'Global Memory access: Each thread fetches 25 pixels directly from high-latency DRAM.'}
                </p>
              </div>
            )}

            {/* Parameter Sliders */}
            <div className="space-y-3 pt-2 border-t border-slate-800">
              {/* Gaussian Sigma */}
              {(algorithm === 'gaussian' || algorithm === 'sharpen' || algorithm === 'full_pipeline') && (
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-slate-400">Gaussian Kernel σ (Sigma):</span>
                    <span className="text-cyan-400 font-bold">{sigma.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="2.5"
                    step="0.1"
                    value={sigma}
                    onChange={(e) => setSigma(parseFloat(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                </div>
              )}

              {/* Sharpening Amount */}
              {(algorithm === 'sharpen' || algorithm === 'full_pipeline') && (
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-slate-400">Unsharp Mask Amount (α):</span>
                    <span className="text-purple-400 font-bold">{amount.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="3.0"
                    step="0.1"
                    value={amount}
                    onChange={(e) => setAmount(parseFloat(e.target.value))}
                    className="w-full accent-purple-500 cursor-pointer"
                  />
                </div>
              )}

              {/* Gamma Exponent */}
              {(algorithm === 'gamma') && (
                <div>
                  <div className="flex justify-between text-xs font-mono mb-1">
                    <span className="text-slate-400">Gamma Exponent (γ):</span>
                    <span className="text-amber-400 font-bold">{gamma.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="2.2"
                    step="0.05"
                    value={gamma}
                    onChange={(e) => setGamma(parseFloat(e.target.value))}
                    className="w-full accent-amber-500 cursor-pointer"
                  />
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="space-y-2 pt-2 border-t border-slate-800">
              <button
                onClick={() => runEnhancement()}
                disabled={isProcessing}
                className="w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-600 hover:from-cyan-400 hover:to-purple-500 text-white font-mono font-bold text-xs shadow-lg shadow-cyan-500/20 transition-all disabled:opacity-50"
              >
                {isProcessing ? (
                  <>
                    <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                    Executing Kernels...
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5 fill-current" />
                    Run Enhancement ({targetDevice.toUpperCase()})
                  </>
                )}
              </button>

              <button
                onClick={() => runEnhancement('full_pipeline')}
                disabled={isProcessing}
                className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 font-mono text-xs transition-all border border-slate-700"
              >
                Run 7-Step Full Pipeline
              </button>
            </div>

            {/* Custom Image Upload Dropzone */}
            <div className="pt-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/png, image/jpeg, image/tiff"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl border border-dashed border-slate-700 hover:border-cyan-500 bg-slate-950/40 text-slate-400 hover:text-cyan-300 text-xs font-mono transition-all"
              >
                <Upload className="w-3.5 h-3.5" />
                Upload Custom Image (PNG/JPG)
              </button>
            </div>
          </div>

          {/* Performance & Memory Timing Card */}
          {result && (
            <div className="p-4 rounded-2xl bg-gradient-to-br from-slate-900/90 to-slate-950/90 border border-slate-800/90 shadow-xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                <span className="text-xs font-mono font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-emerald-400" />
                  Execution Timing Breakdown
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-400 font-semibold border border-emerald-500/30">
                  {result.targetDevice === 'cuda' ? 'GPU Synced' : 'CPU Monotonic'}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Host-to-Device (H2D)</span>
                  <span className="font-semibold text-slate-300">
                    {result.timings.h2d_ms.toFixed(2)} ms
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Kernel Execution</span>
                  <span className="font-semibold text-cyan-400">
                    {result.timings.kernel_ms.toFixed(2)} ms
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Device-to-Host (D2H)</span>
                  <span className="font-semibold text-slate-300">
                    {result.timings.d2h_ms.toFixed(2)} ms
                  </span>
                </div>
                <div className="p-2.5 rounded-xl bg-slate-950/80 border border-slate-800">
                  <span className="text-slate-500 block text-[10px]">Total Latency</span>
                  <span className="font-semibold text-emerald-400">
                    {result.timings.total_ms.toFixed(2)} ms
                  </span>
                </div>
              </div>

              {result.targetDevice === 'cuda' && (
                <div className="p-2 rounded-xl bg-cyan-950/30 border border-cyan-800/40 text-[11px] font-mono text-cyan-300 flex items-center justify-between">
                  <span>DRAM Bandwidth Saved:</span>
                  <strong className="text-white">
                    {result.timings.redundant_reads_avoided_pct}%
                  </strong>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Visualizer & Comparison Display (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {errorMessage && (
            <div className="p-3 rounded-xl bg-red-950/60 border border-red-800/80 text-red-200 text-xs font-mono flex items-center gap-2">
              <ShieldAlert className="w-4 h-4 text-red-400 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}

          {/* Draggable Split Comparison Slider */}
          <div className="p-4 rounded-2xl bg-slate-900/80 border border-slate-800/90 shadow-2xl backdrop-blur-md">
            <ImageComparisonSlider
              originalUrl={originalDataUrl}
              enhancedUrl={result?.outputDataUrl || originalDataUrl}
              width={imageWidth}
              height={imageHeight}
              labelLeft="Original (Grayscale)"
              labelRight={
                result
                  ? `Enhanced (${result.algorithm.replace('_', ' ').toUpperCase()})`
                  : 'Ready (Click Run)'
              }
            />

            {/* Action Bar for Downloads */}
            <div className="flex items-center justify-between mt-3 pt-3 border-t border-slate-800 text-xs font-mono">
              <div className="text-slate-400">
                {result ? (
                  <span className="flex items-center gap-1.5 text-emerald-400">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Processed {result.pixels.toLocaleString()} pixels via {result.targetDevice.toUpperCase()}
                  </span>
                ) : (
                  <span>Select an algorithm and click Run Enhancement</span>
                )}
              </div>

              <div className="flex items-center gap-2">
                {result?.edgeMapDataUrl && (
                  <button
                    onClick={() => handleDownload(result.edgeMapDataUrl!, 'sobel_edge_map.png')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Sobel Edge Map
                  </button>
                )}
                {result && (
                  <button
                    onClick={() => handleDownload(result.outputDataUrl, 'enhanced_output.png')}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-semibold shadow-md shadow-cyan-600/30 transition-all"
                  >
                    <Download className="w-3.5 h-3.5" />
                    Download Enhanced Image
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Dual 256-Bin Intensity Histogram */}
          {result && (
            <HistogramVisualizer
              originalHist={result.histogram.original}
              enhancedHist={result.histogram.enhanced}
            />
          )}
        </div>
      </div>
    </div>
  );
};
