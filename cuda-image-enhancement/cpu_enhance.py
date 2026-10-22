"""
CPU Reference Implementation for Image Enhancement System
Module: cpu_enhance.py
Description: Pure sequential, single-threaded CPU implementations for 5x5 Gaussian
             filtering, 256-bin histogram computation & equalization, unsharp masking,
             gamma correction, Sobel edge detection, and complete pipeline.
Used as the authoritative ground truth for correctness verification and benchmarking.
"""

import numpy as np
import math
import time
from typing import Tuple, Dict, Any


def get_gaussian_kernel_5x5(sigma: float = 1.0) -> np.ndarray:
    """
    Computes a normalized 5x5 Gaussian kernel for the given sigma.
    Center is at (2, 2). Radius is 2.
    """
    kernel = np.zeros((5, 5), dtype=np.float32)
    s = 2.0 * sigma * sigma
    sum_val = 0.0
    for r in range(-2, 3):
        for c in range(-2, 3):
            val = math.exp(-(r * r + c * c) / s) / (math.pi * s)
            kernel[r + 2, c + 2] = val
            sum_val += val
    # Normalize so weights sum to 1.0
    kernel /= sum_val
    return kernel


def cpu_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Applies a 5x5 Gaussian blur sequentially on CPU.
    Handles boundaries using clamp-to-edge (replicate) strategy.
    
    Args:
        image: 2D uint8 numpy array [H, W]
        sigma: Standard deviation of Gaussian kernel (default: 1.0)
    Returns:
        2D uint8 numpy array [H, W]
    """
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)
    kernel = get_gaussian_kernel_5x5(sigma)

    for r in range(H):
        for c in range(W):
            acc = 0.0
            for kr in range(-2, 3):
                for kc in range(-2, 3):
                    # Clamp boundary
                    nr = min(max(r + kr, 0), H - 1)
                    nc = min(max(c + kc, 0), W - 1)
                    acc += float(image[nr, nc]) * kernel[kr + 2, kc + 2]
            output[r, c] = int(round(min(max(acc, 0.0), 255.0)))
    return output


def cpu_histogram(image: np.ndarray) -> np.ndarray:
    """
    Computes 256-bin intensity histogram sequentially on CPU.
    """
    H, W = image.shape
    hist = np.zeros(256, dtype=np.int32)
    for r in range(H):
        for c in range(W):
            pixel = int(image[r, c])
            hist[pixel] += 1
    return hist


def cpu_histogram_equalization(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs full histogram equalization sequentially on CPU.
    
    Returns:
        (equalized_image, hist_original, lut)
    """
    H, W = image.shape
    total_pixels = H * W
    hist = cpu_histogram(image)

    # Compute CDF
    cdf = np.zeros(256, dtype=np.int32)
    acc = 0
    for i in range(256):
        acc += hist[i]
        cdf[i] = acc

    # Find minimum non-zero CDF value
    cdf_min = 0
    for i in range(256):
        if cdf[i] > 0:
            cdf_min = cdf[i]
            break

    # Build 256-entry lookup table
    lut = np.zeros(256, dtype=np.uint8)
    denom = total_pixels - cdf_min
    if denom <= 0:
        # Uniform image fallback
        for i in range(256):
            lut[i] = i
    else:
        for i in range(256):
            val = round(((cdf[i] - cdf_min) / denom) * 255.0)
            lut[i] = min(max(int(val), 0), 255)

    # Apply LUT sequentially
    output = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            output[r, c] = lut[image[r, c]]

    return output, hist, lut


def cpu_unsharp_mask(original: np.ndarray, blurred: np.ndarray, amount: float = 1.5) -> np.ndarray:
    """
    Computes unsharp-mask sharpening on CPU:
    output = original + amount * (original - blurred)
    Clipped to [0, 255].
    """
    H, W = original.shape
    output = np.zeros((H, W), dtype=np.uint8)

    for r in range(H):
        for c in range(W):
            orig_val = float(original[r, c])
            blur_val = float(blurred[r, c])
            sharp_val = orig_val + amount * (orig_val - blur_val)
            output[r, c] = int(round(min(max(sharp_val, 0.0), 255.0)))
    return output


def cpu_gamma_correction(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Applies gamma correction on CPU:
    output = 255 * (input / 255) ** gamma
    """
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)
    inv_255 = 1.0 / 255.0

    # Precompute 256-entry LUT for fast, deterministic evaluation
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = 255.0 * math.pow(i * inv_255, gamma)
        lut[i] = int(round(min(max(val, 0.0), 255.0)))

    for r in range(H):
        for c in range(W):
            output[r, c] = lut[image[r, c]]
    return output


def cpu_sobel_edge(image: np.ndarray) -> np.ndarray:
    """
    Computes Sobel gradient magnitude on CPU:
    Gx = [-1 0 1; -2 0 2; -1 0 1]
    Gy = [-1 -2 -1; 0 0 0; 1 2 1]
    magnitude = sqrt(Gx^2 + Gy^2) clipped to [0, 255]
    """
    H, W = image.shape
    output = np.zeros((H, W), dtype=np.uint8)

    gx_k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    gy_k = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    for r in range(H):
        for c in range(W):
            sum_x = 0.0
            sum_y = 0.0
            for kr in range(-1, 2):
                for kc in range(-1, 2):
                    nr = min(max(r + kr, 0), H - 1)
                    nc = min(max(c + kc, 0), W - 1)
                    val = float(image[nr, nc])
                    sum_x += val * gx_k[kr + 1, kc + 1]
                    sum_y += val * gy_k[kr + 1, kc + 1]
            mag = math.sqrt(sum_x * sum_x + sum_y * sum_y)
            output[r, c] = int(min(max(mag, 0.0), 255.0))
    return output


def cpu_full_pipeline(image: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> Dict[str, Any]:
    """
    Executes the full 7-step enhancement pipeline on CPU:
    1. Grayscale input
    2. Gaussian Denoise (5x5, sigma)
    3. Histogram Equalization
    4. Gaussian Blur for mask
    5. Unsharp-Mask Sharpening
    Returns enhanced image and intermediate stage data with timings.
    """
    t0 = time.perf_counter()

    # Step 1: Denoise
    t_denoise_start = time.perf_counter()
    denoised = cpu_gaussian_blur(image, sigma=sigma)
    t_denoise = (time.perf_counter() - t_denoise_start) * 1000.0

    # Step 2: Hist Eq
    t_eq_start = time.perf_counter()
    equalized, hist, lut = cpu_histogram_equalization(denoised)
    t_eq = (time.perf_counter() - t_eq_start) * 1000.0

    # Step 3: Blur equalized image for unsharp mask
    t_blur_start = time.perf_counter()
    blurred_mask = cpu_gaussian_blur(equalized, sigma=sigma)
    t_blur = (time.perf_counter() - t_blur_start) * 1000.0

    # Step 4: Sharpen
    t_sharp_start = time.perf_counter()
    sharpened = cpu_unsharp_mask(equalized, blurred_mask, amount=amount)
    t_sharp = (time.perf_counter() - t_sharp_start) * 1000.0

    total_time_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "enhanced": sharpened,
        "denoised": denoised,
        "equalized": equalized,
        "histogram": hist,
        "total_ms": total_time_ms,
        "breakdown": {
            "denoise_ms": t_denoise,
            "equalize_ms": t_eq,
            "blur_ms": t_blur,
            "sharpen_ms": t_sharp
        }
    }
