"""
Benchmark Suite for CUDA vs CPU Image Enhancement System
Module: benchmark.py
Description: Evaluates performance across 512x512, 1024x1024, 2048x2048, and 4096x4096
             image resolutions. Performs warm-up runs, collects 5 runs, calculates medians,
             computes speedup, and writes results to results.csv.
"""

import os
import csv
import time
import numpy as np
from typing import List, Dict, Any

from cpu_enhance import cpu_gaussian_blur, cpu_full_pipeline
from cuda_enhance import (
    CUDA_AVAILABLE,
    CUDA_DEVICE_NAME,
    cuda_warmup,
    cuda_gaussian_blur,
    cuda_full_pipeline,
)

BENCHMARK_SIZES = [512, 1024, 2048, 4096]
NUM_RUNS = 5
RESULTS_FILE = "results.csv"


def generate_synthetic_image(H: int, W: int) -> np.ndarray:
    """Generates a realistic synthetic test image with gradients, noise, and sharp edges."""
    rng = np.random.default_rng(42)
    # Background gradient
    y = np.linspace(30, 220, H, dtype=np.float32)[:, None]
    x = np.linspace(30, 220, W, dtype=np.float32)[None, :]
    base = 0.5 * (y + x)
    # Add high frequency synthetic noise
    noise = rng.normal(0, 15, (H, W)).astype(np.float32)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    return img


def run_single_benchmark(H: int, W: int) -> Dict[str, Any]:
    """Runs benchmark for a specific resolution."""
    print(f"\n[+] Running Benchmark for {W}x{H} ({H * W:,} pixels)...")
    img = generate_synthetic_image(H, W)

    # 1. CPU Gaussian Blur (5 runs median)
    cpu_gauss_times = []
    print("  -> Benchmarking CPU Gaussian 5x5...")
    # Warmup
    _ = cpu_gaussian_blur(img[:64, :64])
    for i in range(NUM_RUNS):
        t0 = time.perf_counter()
        _ = cpu_gaussian_blur(img)
        el = (time.perf_counter() - t0) * 1000.0
        cpu_gauss_times.append(el)
    cpu_gauss_median = float(np.median(cpu_gauss_times))
    print(f"     CPU Gaussian Median: {cpu_gauss_median:.2f} ms")

    # 2. CPU Full Pipeline (5 runs median)
    cpu_pipeline_times = []
    print("  -> Benchmarking CPU Full Pipeline...")
    for i in range(NUM_RUNS):
        res = cpu_full_pipeline(img)
        cpu_pipeline_times.append(res["total_ms"])
    cpu_pipe_median = float(np.median(cpu_pipeline_times))
    print(f"     CPU Pipeline Median: {cpu_pipe_median:.2f} ms")

    row: Dict[str, Any] = {
        "Resolution": f"{W}x{H}",
        "Pixels": H * W,
        "CPU_Gauss_ms": round(cpu_gauss_median, 3),
        "CPU_Pipeline_ms": round(cpu_pipe_median, 3),
        "CUDA_Global_Gauss_ms": "N/A",
        "CUDA_Global_Kernel_ms": "N/A",
        "CUDA_Shared_Gauss_ms": "N/A",
        "CUDA_Shared_Kernel_ms": "N/A",
        "CUDA_Pipeline_ms": "N/A",
        "CUDA_Pipeline_Kernel_ms": "N/A",
        "CUDA_H2D_ms": "N/A",
        "CUDA_D2H_ms": "N/A",
        "Gauss_Speedup_Shared": "N/A",
        "Pipeline_Speedup": "N/A",
        "Device": CUDA_DEVICE_NAME if CUDA_AVAILABLE else "CPU Fallback (No CUDA GPU)"
    }

    if CUDA_AVAILABLE:
        # CUDA Warmup
        cuda_warmup()

        # CUDA Global Gaussian
        cuda_global_total = []
        cuda_global_k = []
        for i in range(NUM_RUNS):
            res = cuda_gaussian_blur(img, use_shared_memory=False)
            cuda_global_total.append(res["total_ms"])
            cuda_global_k.append(res["kernel_ms"])
        c_glob_tot = float(np.median(cuda_global_total))
        c_glob_k = float(np.median(cuda_global_k))

        # CUDA Shared Gaussian
        cuda_shared_total = []
        cuda_shared_k = []
        h2d_list = []
        d2h_list = []
        for i in range(NUM_RUNS):
            res = cuda_gaussian_blur(img, use_shared_memory=True)
            cuda_shared_total.append(res["total_ms"])
            cuda_shared_k.append(res["kernel_ms"])
            h2d_list.append(res["h2d_ms"])
            d2h_list.append(res["d2h_ms"])
        c_sh_tot = float(np.median(cuda_shared_total))
        c_sh_k = float(np.median(cuda_shared_k))
        h2d_med = float(np.median(h2d_list))
        d2h_med = float(np.median(d2h_list))

        # CUDA Full Pipeline
        pipe_total = []
        pipe_k = []
        for i in range(NUM_RUNS):
            res = cuda_full_pipeline(img, use_shared_memory=True)
            pipe_total.append(res["total_ms"])
            pipe_k.append(res["kernel_ms"])
        c_pipe_tot = float(np.median(pipe_total))
        c_pipe_k = float(np.median(pipe_k))

        # Speedups
        gauss_speedup = cpu_gauss_median / c_sh_tot if c_sh_tot > 0 else 0.0
        pipe_speedup = cpu_pipe_median / c_pipe_tot if c_pipe_tot > 0 else 0.0

        row["CUDA_Global_Gauss_ms"] = round(c_glob_tot, 3)
        row["CUDA_Global_Kernel_ms"] = round(c_glob_k, 3)
        row["CUDA_Shared_Gauss_ms"] = round(c_sh_tot, 3)
        row["CUDA_Shared_Kernel_ms"] = round(c_sh_k, 3)
        row["CUDA_Pipeline_ms"] = round(c_pipe_tot, 3)
        row["CUDA_Pipeline_Kernel_ms"] = round(c_pipe_k, 3)
        row["CUDA_H2D_ms"] = round(h2d_med, 3)
        row["CUDA_D2H_ms"] = round(d2h_med, 3)
        row["Gauss_Speedup_Shared"] = round(gauss_speedup, 2)
        row["Pipeline_Speedup"] = round(pipe_speedup, 2)

        print(f"     CUDA Shared Gaussian: {c_sh_tot:.2f} ms (Kernel: {c_sh_k:.2f} ms, Speedup: {gauss_speedup:.1f}x)")
        print(f"     CUDA Full Pipeline: {c_pipe_tot:.2f} ms (Kernel: {c_pipe_k:.2f} ms, Speedup: {pipe_speedup:.1f}x)")
    else:
        print("     [Note] CUDA GPU is not available on this host. Skipping GPU execution; CPU results recorded.")

    return row


def save_results_csv(records: List[Dict[str, Any]], filepath: str = RESULTS_FILE):
    """Saves benchmark results table to CSV format."""
    if not records:
        return
    fieldnames = list(records[0].keys())
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[✓] Benchmark results successfully saved to {filepath}")


def main():
    print("=" * 70)
    print("CUDA-ACCELERATED IMAGE ENHANCEMENT - BENCHMARK SUITE")
    print(f"Execution Target: {'NVIDIA ' + CUDA_DEVICE_NAME if CUDA_AVAILABLE else 'CPU Fallback'}")
    print("=" * 70)

    results = []
    # Test through resolutions
    for s in BENCHMARK_SIZES:
        rec = run_single_benchmark(s, s)
        results.append(rec)

    save_results_csv(results, RESULTS_FILE)


if __name__ == "__main__":
    main()
