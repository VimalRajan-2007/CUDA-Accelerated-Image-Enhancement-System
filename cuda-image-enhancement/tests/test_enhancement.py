"""
Unit Tests for CUDA and CPU Enhancement Algorithms
Module: tests/test_enhancement.py
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cpu_enhance import (
    get_gaussian_kernel_5x5,
    cpu_gaussian_blur,
    cpu_histogram,
    cpu_histogram_equalization,
    cpu_unsharp_mask,
    cpu_gamma_correction,
    cpu_sobel_edge,
    cpu_full_pipeline
)
from cuda_enhance import (
    CUDA_AVAILABLE,
    cuda_gaussian_blur,
    cuda_histogram_equalization,
    cuda_unsharp_mask,
    cuda_gamma_correction,
    cuda_sobel
)


class TestImageEnhancement(unittest.TestCase):

    def setUp(self):
        # 64x64 synthetic test image
        rng = np.random.default_rng(42)
        self.img = rng.integers(0, 256, (64, 64), dtype=np.uint8)
        # Non-multiple boundary test image: 37x53
        self.odd_img = rng.integers(0, 256, (37, 53), dtype=np.uint8)

    def test_gaussian_kernel_normalization(self):
        k = get_gaussian_kernel_5x5(sigma=1.0)
        self.assertEqual(k.shape, (5, 5))
        self.assertAlmostEqual(float(np.sum(k)), 1.0, places=5)

    def test_cpu_histogram(self):
        hist = cpu_histogram(self.img)
        self.assertEqual(len(hist), 256)
        self.assertEqual(int(np.sum(hist)), 64 * 64)

    def test_cpu_gaussian_blur_bounds(self):
        out = cpu_gaussian_blur(self.img, sigma=1.0)
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(np.all(out >= 0) and np.all(out <= 255))

    def test_cpu_odd_dimensions_boundary(self):
        out = cpu_gaussian_blur(self.odd_img, sigma=1.0)
        self.assertEqual(out.shape, (37, 53))

    def test_cpu_histogram_equalization(self):
        eq, hist, lut = cpu_histogram_equalization(self.img)
        self.assertEqual(eq.shape, self.img.shape)
        self.assertEqual(len(lut), 256)

    def test_cpu_unsharp_mask(self):
        blurred = cpu_gaussian_blur(self.img, sigma=1.0)
        sharp = cpu_unsharp_mask(self.img, blurred, amount=1.5)
        self.assertEqual(sharp.shape, self.img.shape)
        self.assertTrue(np.all(sharp >= 0) and np.all(sharp <= 255))

    def test_cpu_gamma_correction(self):
        darker = cpu_gamma_correction(self.img, gamma=1.8)
        brighter = cpu_gamma_correction(self.img, gamma=0.5)
        self.assertEqual(darker.shape, self.img.shape)
        self.assertEqual(brighter.shape, self.img.shape)

    def test_cpu_sobel_edge(self):
        sobel = cpu_sobel_edge(self.img)
        self.assertEqual(sobel.shape, self.img.shape)

    def test_cpu_full_pipeline(self):
        res = cpu_full_pipeline(self.img)
        self.assertIn("enhanced", res)
        self.assertIn("total_ms", res)
        self.assertGreater(res["total_ms"], 0.0)

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA GPU not available on this host")
    def test_cuda_vs_cpu_gaussian(self):
        cpu_out = cpu_gaussian_blur(self.img, sigma=1.0)
        gpu_res = cuda_gaussian_blur(self.img, sigma=1.0, use_shared_memory=True)
        gpu_out = gpu_res["output"]
        diff = np.abs(cpu_out.astype(np.int32) - gpu_out.astype(np.int32))
        max_diff = np.max(diff)
        self.assertLessEqual(max_diff, 1)


if __name__ == "__main__":
    unittest.main()
