"""
CUDA Implementation of Image Enhancement Algorithms
Module: cuda_enhance.py
Description: Genuine Numba CUDA kernels for:
  1. 5x5 Gaussian blur (Global memory)
  2. 5x5 Gaussian blur (Shared-memory 20x20 tiling with 2-pixel halo cooperative loading)
  3. 256-bin Histogram computation (Shared-memory block histograms + atomic adds)
  4. Histogram equalization LUT remapping
  5. Unsharp-mask sharpening
  6. Gamma correction
  7. Sobel edge detection
  8. Full 7-stage enhancement pipeline

Handles partial blocks, grid calculation, device memory transfers, and timing synchronization.
"""

import math
import time
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Probe CUDA capability via Numba
CUDA_AVAILABLE = False
CUDA_DEVICE_NAME = "None"
CUDA_CAPABILITY = "N/A"
CUDA_TOTAL_MEMORY_MB = 0
CUDA_INIT_ERROR = ""

try:
    from numba import cuda, float32, int32, uint8
    if cuda.is_available():
        device = cuda.get_current_device()
        CUDA_AVAILABLE = True
        CUDA_DEVICE_NAME = device.name.decode("utf-8") if isinstance(device.name, bytes) else str(device.name)
        CUDA_CAPABILITY = f"{device.compute_capability[0]}.{device.compute_capability[1]}"
        # Memory info if available
        try:
            mem_info = cuda.current_context().get_memory_info()
            CUDA_TOTAL_MEMORY_MB = round(mem_info.total / (1024 * 1024))
        except Exception:
            CUDA_TOTAL_MEMORY_MB = 0
    else:
        CUDA_INIT_ERROR = "NVIDIA CUDA driver or compatible GPU not detected by Numba."
except Exception as e:
    CUDA_INIT_ERROR = f"Failed to initialize Numba CUDA: {str(e)}"


# ============================================================================
# CONSTANTS & PRECOMPUTED WEIGHTS
# ============================================================================

def get_gaussian_weights_5x5(sigma: float = 1.0) -> np.ndarray:
    """Precompute normalized 5x5 float32 Gaussian weights."""
    k = np.zeros((5, 5), dtype=np.float32)
    s = 2.0 * sigma * sigma
    sum_val = 0.0
    for r in range(-2, 3):
        for c in range(-2, 3):
            val = math.exp(-(r * r + c * c) / s) / (math.pi * s)
            k[r + 2, c + 2] = val
            sum_val += val
    k /= sum_val
    return k


# ============================================================================
# CUDA KERNEL DEFINITIONS (Compiled when CUDA is available)
# ============================================================================

if CUDA_AVAILABLE:
    # ------------------------------------------------------------------------
    # 1. Global Memory 5x5 Gaussian Filter Kernel
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_gaussian_global(d_in, d_out, d_weights, H, W):
        r, c = cuda.grid(2)
        if r < H and c < W:
            acc = 0.0
            for kr in range(-2, 3):
                for kc in range(-2, 3):
                    # Clamp-to-edge boundary condition
                    nr = min(max(r + kr, 0), H - 1)
                    nc = min(max(c + kc, 0), W - 1)
                    acc += float32(d_in[nr, nc]) * d_weights[kr + 2, kc + 2]
            # Clamp and round to uint8
            if acc < 0.0:
                acc = 0.0
            elif acc > 255.0:
                acc = 255.0
            d_out[r, c] = uint8(acc + 0.5)

    # ------------------------------------------------------------------------
    # 2. Shared-Memory Tiled 5x5 Gaussian Filter Kernel
    # Tile parameters: 16x16 output tile, 20x20 shared memory tile (2-pixel halo)
    # Total shared memory elements = 20 * 20 = 400
    # Block dimensions = 16 x 16 = 256 threads
    # Cooperative loading: 256 threads load first 256 elements;
    # 144 threads load the remaining 144 elements (400 total).
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_gaussian_shared(d_in, d_out, d_weights, H, W):
        # Allocate 20x20 shared memory tile
        s_tile = cuda.shared.array(shape=(20, 20), dtype=float32)

        tx = cuda.threadIdx.x  # 0 .. 15
        ty = cuda.threadIdx.y  # 0 .. 15
        tid = ty * 16 + tx      # Linear thread ID: 0 .. 255

        # Block top-left corner in global coordinates (offset by -2 halo)
        block_origin_r = cuda.blockIdx.y * 16 - 2
        block_origin_c = cuda.blockIdx.x * 16 - 2

        # Pass 1: Each thread loads one element (0 .. 255)
        if tid < 256:
            tile_r = tid // 20
            tile_c = tid % 20
            gr = min(max(block_origin_r + tile_r, 0), H - 1)
            gc = min(max(block_origin_c + tile_c, 0), W - 1)
            s_tile[tile_r, tile_c] = float32(d_in[gr, gc])

        # Pass 2: First 144 threads load remaining elements (256 .. 399)
        if tid < 144:
            tid2 = tid + 256
            tile_r2 = tid2 // 20
            tile_c2 = tid2 % 20
            gr2 = min(max(block_origin_r + tile_r2, 0), H - 1)
            gc2 = min(max(block_origin_c + tile_c2, 0), W - 1)
            s_tile[tile_r2, tile_c2] = float32(d_in[gr2, gc2])

        # Synchronize so all shared memory loads are visible
        cuda.syncthreads()

        # Compute output for thread's pixel
        r = cuda.blockIdx.y * 16 + ty
        c = cuda.blockIdx.x * 16 + tx

        if r < H and c < W:
            acc = 0.0
            # Center of the filter in shared tile is at (ty + 2, tx + 2)
            sr_center = ty + 2
            sc_center = tx + 2
            for kr in range(-2, 3):
                for kc in range(-2, 3):
                    acc += s_tile[sr_center + kr, sc_center + kc] * d_weights[kr + 2, kc + 2]

            if acc < 0.0:
                acc = 0.0
            elif acc > 255.0:
                acc = 255.0
            d_out[r, c] = uint8(acc + 0.5)

    # ------------------------------------------------------------------------
    # 3. Shared-Memory Histogram Kernel
    # 256 bins for 8-bit image. Uses shared memory histogram per block
    # and merges to global histogram with atomic additions.
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_histogram_shared(d_in, d_global_hist, H, W):
        s_hist = cuda.shared.array(shape=256, dtype=int32)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        tid = ty * 16 + tx

        # Initialize shared memory histogram bins to 0
        if tid < 256:
            s_hist[tid] = 0
        cuda.syncthreads()

        # Accumulate local histogram in shared memory
        r, c = cuda.grid(2)
        if r < H and c < W:
            val = int32(d_in[r, c])
            cuda.atomic.add(s_hist, val, 1)

        cuda.syncthreads()

        # Merge local block histogram into global memory histogram
        if tid < 256:
            cuda.atomic.add(d_global_hist, tid, s_hist[tid])

    # ------------------------------------------------------------------------
    # 4. Lookup Table (LUT) Remapping Kernel
    # Parallel pixel remapping for Histogram Equalization and Gamma Correction
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_apply_lut(d_in, d_out, d_lut, H, W):
        r, c = cuda.grid(2)
        if r < H and c < W:
            val = d_in[r, c]
            d_out[r, c] = d_lut[val]

    # ------------------------------------------------------------------------
    # 5. Unsharp-Mask Sharpening Kernel
    # output = original + amount * (original - blurred)
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_unsharp_mask(d_orig, d_blur, d_out, amount, H, W):
        r, c = cuda.grid(2)
        if r < H and c < W:
            orig = float32(d_orig[r, c])
            blur = float32(d_blur[r, c])
            sharp = orig + amount * (orig - blur)
            if sharp < 0.0:
                sharp = 0.0
            elif sharp > 255.0:
                sharp = 255.0
            d_out[r, c] = uint8(sharp + 0.5)

    # ------------------------------------------------------------------------
    # 6. Sobel Edge Detection Kernel
    # 3x3 Gx and Gy convolution and magnitude computation
    # ------------------------------------------------------------------------
    @cuda.jit
    def kernel_sobel(d_in, d_out, H, W):
        r, c = cuda.grid(2)
        if r < H and c < W:
            # 3x3 window with clamped borders
            r_prev = max(r - 1, 0)
            r_next = min(r + 1, H - 1)
            c_prev = max(c - 1, 0)
            c_next = min(c + 1, W - 1)

            p00 = float32(d_in[r_prev, c_prev])
            p01 = float32(d_in[r_prev, c])
            p02 = float32(d_in[r_prev, c_next])

            p10 = float32(d_in[r, c_prev])
            # p11 = center
            p12 = float32(d_in[r, c_next])

            p20 = float32(d_in[r_next, c_prev])
            p21 = float32(d_in[r_next, c])
            p22 = float32(d_in[r_next, c_next])

            gx = (-1.0 * p00) + (1.0 * p02) + (-2.0 * p10) + (2.0 * p12) + (-1.0 * p20) + (1.0 * p22)
            gy = (-1.0 * p00) + (-2.0 * p01) + (-1.0 * p02) + (1.0 * p20) + (2.0 * p21) + (1.0 * p22)

            mag = math.sqrt(gx * gx + gy * gy)
            if mag > 255.0:
                mag = 255.0
            d_out[r, c] = uint8(mag)


# ============================================================================
# HELPER FUNCTIONS & GRID COMPUTATION
# ============================================================================

def get_grid_dims(H: int, W: int, block_dim: Tuple[int, int] = (16, 16)) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Computes (blockspergrid, threadsperblock) covering full image H x W.
    Handles non-multiples and partial blocks safely.
    """
    tx, ty = block_dim
    bx = math.ceil(W / tx)
    by = math.ceil(H / ty)
    return (bx, by), (tx, ty)


def cuda_warmup():
    """
    Performs a minimal dummy kernel launch to trigger JIT compilation
    so that benchmark timings measure actual runtime without compilation overhead.
    """
    if not CUDA_AVAILABLE:
        return
    try:
        dummy = np.zeros((32, 32), dtype=np.uint8)
        d_dummy = cuda.to_device(dummy)
        d_out = cuda.device_array_like(dummy)
        weights = get_gaussian_weights_5x5(1.0)
        d_w = cuda.to_device(weights)
        grid, block = get_grid_dims(32, 32)
        kernel_gaussian_global[grid, block](d_dummy, d_out, d_w, 32, 32)
        cuda.synchronize()
    except Exception:
        pass


# ============================================================================
# HIGH-LEVEL CALLABLE CUDA OPERATIONS WITH ACCURATE TIMINGS
# ============================================================================

def cuda_gaussian_blur(image: np.ndarray, sigma: float = 1.0, use_shared_memory: bool = True) -> Dict[str, Any]:
    """
    Executes Gaussian blur on GPU using either global or shared memory kernel.
    Returns:
        output array, h2d_ms, kernel_ms, d2h_ms, total_ms
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = image.shape
    weights = get_gaussian_weights_5x5(sigma)

    grid, block = get_grid_dims(H, W)

    # 1. Host-to-Device transfer
    cuda.synchronize()
    t_h2d_start = time.perf_counter()
    d_in = cuda.to_device(image)
    d_out = cuda.device_array((H, W), dtype=np.uint8)
    d_weights = cuda.to_device(weights)
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t_h2d_start) * 1000.0

    # 2. Kernel execution
    cuda.synchronize()
    t_kernel_start = time.perf_counter()
    if use_shared_memory:
        kernel_gaussian_shared[grid, block](d_in, d_out, d_weights, H, W)
    else:
        kernel_gaussian_global[grid, block](d_in, d_out, d_weights, H, W)
    cuda.synchronize()
    t_kernel = (time.perf_counter() - t_kernel_start) * 1000.0

    # 3. Device-to-Host transfer
    t_d2h_start = time.perf_counter()
    output = d_out.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d2h_start) * 1000.0

    total_ms = t_h2d + t_kernel + t_d2h

    return {
        "output": output,
        "h2d_ms": t_h2d,
        "kernel_ms": t_kernel,
        "d2h_ms": t_d2h,
        "total_ms": total_ms,
        "mode": "shared_memory" if use_shared_memory else "global_memory"
    }


def cuda_histogram_equalization(image: np.ndarray) -> Dict[str, Any]:
    """
    Executes histogram computation on GPU, builds CDF/LUT, and remaps on GPU.
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = image.shape
    grid, block = get_grid_dims(H, W)

    # Transfer image
    cuda.synchronize()
    t_start = time.perf_counter()
    d_in = cuda.to_device(image)
    d_hist = cuda.to_device(np.zeros(256, dtype=np.int32))
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t_start) * 1000.0

    # Compute Histogram Kernel
    t_k_start = time.perf_counter()
    kernel_histogram_shared[grid, block](d_in, d_hist, H, W)
    cuda.synchronize()
    t_k_hist = (time.perf_counter() - t_k_start) * 1000.0

    # Retrieve histogram for CDF computation
    hist = d_hist.copy_to_host()

    # Build CDF & LUT on host (O(256) overhead is ~0.005 ms)
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)][0] if np.any(cdf > 0) else 0
    total_pixels = H * W
    denom = total_pixels - cdf_min

    lut = np.zeros(256, dtype=np.uint8)
    if denom <= 0:
        lut = np.arange(256, dtype=np.uint8)
    else:
        for i in range(256):
            v = round(((cdf[i] - cdf_min) / denom) * 255.0)
            lut[i] = min(max(int(v), 0), 255)

    # Transfer LUT and remap pixels in parallel on GPU
    d_lut = cuda.to_device(lut)
    d_out = cuda.device_array((H, W), dtype=np.uint8)

    t_k2_start = time.perf_counter()
    kernel_apply_lut[grid, block](d_in, d_out, d_lut, H, W)
    cuda.synchronize()
    t_k_lut = (time.perf_counter() - t_k2_start) * 1000.0

    # Copy output
    t_d2h_start = time.perf_counter()
    output = d_out.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d2h_start) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0

    return {
        "output": output,
        "histogram": hist,
        "lut": lut,
        "h2d_ms": t_h2d,
        "kernel_ms": t_k_hist + t_k_lut,
        "d2h_ms": t_d2h,
        "total_ms": total_ms
    }


def cuda_unsharp_mask(original: np.ndarray, blurred: np.ndarray, amount: float = 1.5) -> Dict[str, Any]:
    """Unsharp masking on GPU."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = original.shape
    grid, block = get_grid_dims(H, W)

    cuda.synchronize()
    t0 = time.perf_counter()
    d_orig = cuda.to_device(original)
    d_blur = cuda.to_device(blurred)
    d_out = cuda.device_array((H, W), dtype=np.uint8)
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t0) * 1000.0

    t_k0 = time.perf_counter()
    kernel_unsharp_mask[grid, block](d_orig, d_blur, d_out, float(amount), H, W)
    cuda.synchronize()
    t_k = (time.perf_counter() - t_k0) * 1000.0

    t_d0 = time.perf_counter()
    output = d_out.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d0) * 1000.0

    return {
        "output": output,
        "h2d_ms": t_h2d,
        "kernel_ms": t_k,
        "d2h_ms": t_d2h,
        "total_ms": t_h2d + t_k + t_d2h
    }


def cuda_gamma_correction(image: np.ndarray, gamma: float = 0.6) -> Dict[str, Any]:
    """Gamma correction via GPU LUT remapping."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = image.shape
    grid, block = get_grid_dims(H, W)

    # 256-entry gamma LUT
    inv_255 = 1.0 / 255.0
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        val = 255.0 * math.pow(i * inv_255, gamma)
        lut[i] = int(round(min(max(val, 0.0), 255.0)))

    cuda.synchronize()
    t0 = time.perf_counter()
    d_in = cuda.to_device(image)
    d_lut = cuda.to_device(lut)
    d_out = cuda.device_array((H, W), dtype=np.uint8)
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t0) * 1000.0

    t_k0 = time.perf_counter()
    kernel_apply_lut[grid, block](d_in, d_out, d_lut, H, W)
    cuda.synchronize()
    t_k = (time.perf_counter() - t_k0) * 1000.0

    t_d0 = time.perf_counter()
    output = d_out.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d0) * 1000.0

    return {
        "output": output,
        "h2d_ms": t_h2d,
        "kernel_ms": t_k,
        "d2h_ms": t_d2h,
        "total_ms": t_h2d + t_k + t_d2h
    }


def cuda_sobel(image: np.ndarray) -> Dict[str, Any]:
    """Sobel edge detection on GPU."""
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = image.shape
    grid, block = get_grid_dims(H, W)

    cuda.synchronize()
    t0 = time.perf_counter()
    d_in = cuda.to_device(image)
    d_out = cuda.device_array((H, W), dtype=np.uint8)
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t0) * 1000.0

    t_k0 = time.perf_counter()
    kernel_sobel[grid, block](d_in, d_out, H, W)
    cuda.synchronize()
    t_k = (time.perf_counter() - t_k0) * 1000.0

    t_d0 = time.perf_counter()
    output = d_out.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d0) * 1000.0

    return {
        "output": output,
        "h2d_ms": t_h2d,
        "kernel_ms": t_k,
        "d2h_ms": t_d2h,
        "total_ms": t_h2d + t_k + t_d2h
    }


def cuda_full_pipeline(image: np.ndarray, sigma: float = 1.0, amount: float = 1.5, use_shared_memory: bool = True) -> Dict[str, Any]:
    """
    Executes the complete 7-step enhancement pipeline on GPU with intermediate buffers kept in device VRAM:
      1. Host-to-Device transfer of input image
      2. Denoise with 5x5 Gaussian (Shared or Global) -> d_denoised
      3. Histogram computation -> d_hist
      4. CDF & LUT generation on host
      5. Remap d_denoised -> d_equalized
      6. Blur d_equalized -> d_mask
      7. Unsharp mask: d_equalized + amount * (d_equalized - d_mask) -> d_final
      8. Device-to-Host transfer of enhanced image
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA is not available on this system.")

    H, W = image.shape
    grid, block = get_grid_dims(H, W)
    weights = get_gaussian_weights_5x5(sigma)

    cuda.synchronize()
    t_total_start = time.perf_counter()

    # 1. H2D
    t_h2d_start = time.perf_counter()
    d_in = cuda.to_device(image)
    d_weights = cuda.to_device(weights)
    d_denoised = cuda.device_array((H, W), dtype=np.uint8)
    d_equalized = cuda.device_array((H, W), dtype=np.uint8)
    d_mask = cuda.device_array((H, W), dtype=np.uint8)
    d_final = cuda.device_array((H, W), dtype=np.uint8)
    d_hist = cuda.to_device(np.zeros(256, dtype=np.int32))
    cuda.synchronize()
    t_h2d = (time.perf_counter() - t_h2d_start) * 1000.0

    # 2. Gaussian Denoise Kernel
    t_k_start = time.perf_counter()
    if use_shared_memory:
        kernel_gaussian_shared[grid, block](d_in, d_denoised, d_weights, H, W)
    else:
        kernel_gaussian_global[grid, block](d_in, d_denoised, d_weights, H, W)
    cuda.synchronize()

    # 3. Histogram Kernel
    kernel_histogram_shared[grid, block](d_denoised, d_hist, H, W)
    cuda.synchronize()
    hist = d_hist.copy_to_host()

    # 4. CDF & LUT
    cdf = np.cumsum(hist)
    cdf_min = cdf[np.nonzero(cdf)][0] if np.any(cdf > 0) else 0
    total_pixels = H * W
    denom = total_pixels - cdf_min

    lut = np.zeros(256, dtype=np.uint8)
    if denom <= 0:
        lut = np.arange(256, dtype=np.uint8)
    else:
        for i in range(256):
            v = round(((cdf[i] - cdf_min) / denom) * 255.0)
            lut[i] = min(max(int(v), 0), 255)
    d_lut = cuda.to_device(lut)

    # 5. LUT Remapping Kernel
    kernel_apply_lut[grid, block](d_denoised, d_equalized, d_lut, H, W)
    cuda.synchronize()

    # 6. Blur Equalized image for Unsharp Mask
    if use_shared_memory:
        kernel_gaussian_shared[grid, block](d_equalized, d_mask, d_weights, H, W)
    else:
        kernel_gaussian_global[grid, block](d_equalized, d_mask, d_weights, H, W)
    cuda.synchronize()

    # 7. Unsharp Mask Kernel
    kernel_unsharp_mask[grid, block](d_equalized, d_mask, d_final, float(amount), H, W)
    cuda.synchronize()
    t_kernel_total = (time.perf_counter() - t_k_start) * 1000.0

    # 8. D2H
    t_d2h_start = time.perf_counter()
    final_output = d_final.copy_to_host()
    cuda.synchronize()
    t_d2h = (time.perf_counter() - t_d2h_start) * 1000.0

    total_ms = (time.perf_counter() - t_total_start) * 1000.0

    return {
        "output": final_output,
        "histogram": hist,
        "h2d_ms": t_h2d,
        "kernel_ms": t_kernel_total,
        "d2h_ms": t_d2h,
        "total_ms": total_ms
    }
