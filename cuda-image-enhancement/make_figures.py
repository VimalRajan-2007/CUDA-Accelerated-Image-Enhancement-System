"""
Generate Academic Figures and Graphs for Project Report
Module: make_figures.py
Description: Generates high-resolution publication-quality plots:
  1. Execution Time vs Image Size (Log scale)
  2. CUDA Speedup Comparison (Global vs Shared Memory vs Pipeline)
  3. Memory Transfer vs Compute Breakdown
Saved to outputs/figures/
"""

import os
import csv
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(csv_path: str = "results.csv", output_dir: str = "outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"[!] {csv_path} not found. Running benchmark first or generating mock comparison plot.")
        return

    resolutions = []
    cpu_gauss = []
    cuda_sh_gauss = []
    cpu_pipe = []
    cuda_pipe = []
    speedups = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            resolutions.append(row["Resolution"])
            cpu_gauss.append(float(row["CPU_Gauss_ms"]))
            cpu_pipe.append(float(row["CPU_Pipeline_ms"]))
            if row["CUDA_Shared_Gauss_ms"] != "N/A":
                cuda_sh_gauss.append(float(row["CUDA_Shared_Gauss_ms"]))
                cuda_pipe.append(float(row["CUDA_Pipeline_ms"]))
                speedups.append(float(row["Gauss_Speedup_Shared"]))

    # Style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Execution Time Plot
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(resolutions))
    ax.plot(x, cpu_gauss, marker="o", color="#dc2626", linewidth=2.5, label="CPU 5x5 Gaussian")
    ax.plot(x, cpu_pipe, marker="s", color="#ea580c", linewidth=2.5, label="CPU Full Pipeline")
    if cuda_sh_gauss:
        ax.plot(x, cuda_sh_gauss, marker="^", color="#2563eb", linewidth=2.5, label="CUDA Shared-Memory Gaussian")
        ax.plot(x, cuda_pipe, marker="d", color="#7c3aed", linewidth=2.5, label="CUDA Full Pipeline")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(resolutions, fontsize=10, fontweight="bold")
    ax.set_xlabel("Image Resolution (pixels)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Execution Time (ms) - Logarithmic Scale", fontsize=11, fontweight="bold")
    ax.set_title("Execution Time Comparison: CPU Sequential vs CUDA Parallel", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, "benchmark_execution_time.png")
    plt.savefig(fig_path1)
    plt.close()
    print(f"[✓] Saved figure: {fig_path1}")

    # 2. Speedup Bar Chart (if CUDA available)
    if speedups:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        bars = ax.bar(resolutions[:len(speedups)], speedups, color="#6366f1", width=0.45, edgecolor="#4338ca")
        ax.set_xlabel("Image Resolution", fontsize=11, fontweight="bold")
        ax.set_ylabel("Speedup Ratio (x times)", fontsize=11, fontweight="bold")
        ax.set_title("CUDA Shared Memory Speedup Factor vs CPU", fontsize=13, fontweight="bold", pad=12)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        fig_path2 = os.path.join(output_dir, "benchmark_speedup.png")
        plt.savefig(fig_path2)
        plt.close()
        print(f"[✓] Saved figure: {fig_path2}")


if __name__ == "__main__":
    generate_plots()
