"""
Flask REST API and Application Server
Module: app.py
Description: Production-ready Flask REST backend for CUDA-Accelerated Image Enhancement System.
             Endpoints:
               GET  /api/status
               POST /api/upload
               POST /api/enhance
               POST /api/benchmark
               POST /api/verify
               GET  /api/results
               GET  /api/download/<filename>
               GET  /
"""

import os
import io
import time
import uuid
import base64
import csv
from typing import Dict, Any
from flask import Flask, request, jsonify, send_from_directory, render_template
from PIL import Image
import numpy as np

# Import algorithmic backends
from cpu_enhance import (
    cpu_gaussian_blur,
    cpu_histogram_equalization,
    cpu_unsharp_mask,
    cpu_gamma_correction,
    cpu_sobel_edge,
    cpu_full_pipeline,
    cpu_histogram
)
from cuda_enhance import (
    CUDA_AVAILABLE,
    CUDA_DEVICE_NAME,
    CUDA_CAPABILITY,
    CUDA_TOTAL_MEMORY_MB,
    CUDA_INIT_ERROR,
    cuda_gaussian_blur,
    cuda_histogram_equalization,
    cuda_unsharp_mask,
    cuda_gamma_correction,
    cuda_sobel,
    cuda_full_pipeline,
)
from benchmark import run_single_benchmark, save_results_csv, RESULTS_FILE
from verify_kernels import run_verification_suite

app = Flask(__name__, template_folder="templates", static_folder="static")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tif", "tiff"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def numpy_to_base64_png(arr: np.ndarray) -> str:
    """Converts 2D uint8 numpy array to base64 data URI."""
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


# ----------------------------------------------------------------------------
# 1. API: System Status & GPU Detection
# ----------------------------------------------------------------------------
@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "online",
        "cuda_available": CUDA_AVAILABLE,
        "execution_mode": "cuda_gpu" if CUDA_AVAILABLE else "cpu_fallback",
        "gpu_name": CUDA_DEVICE_NAME,
        "cuda_capability": CUDA_CAPABILITY,
        "total_memory_mb": CUDA_TOTAL_MEMORY_MB,
        "init_message": "NVIDIA GPU ready for CUDA acceleration" if CUDA_AVAILABLE else (CUDA_INIT_ERROR or "CPU fallback active"),
        "timestamp": time.time()
    })


# ----------------------------------------------------------------------------
# 2. API: Image Upload
# ----------------------------------------------------------------------------
@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    try:
        # Load and validate image with PIL
        pil_img = Image.open(file.stream)
        # Convert to 8-bit grayscale
        gray_img = pil_img.convert("L")
        arr = np.array(gray_img, dtype=np.uint8)
        H, W = arr.shape

        file_id = f"img_{uuid.uuid4().hex[:10]}"
        filename = f"{file_id}.png"
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        gray_img.save(save_path, format="PNG")

        hist = cpu_histogram(arr).tolist()
        b64_preview = numpy_to_base64_png(arr)

        return jsonify({
            "message": "Image uploaded successfully",
            "file_id": file_id,
            "filename": filename,
            "width": W,
            "height": H,
            "pixels": H * W,
            "histogram": hist,
            "preview_url": b64_preview
        })
    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


# ----------------------------------------------------------------------------
# 3. API: Enhance Image
# ----------------------------------------------------------------------------
@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")
    algorithm = data.get("algorithm", "full_pipeline")
    target_device = data.get("target_device", "auto")  # "cuda", "cpu", "auto"
    use_shared_memory = bool(data.get("use_shared_memory", True))
    sigma = float(data.get("sigma", 1.0))
    amount = float(data.get("amount", 1.5))
    gamma = float(data.get("gamma", 0.6))

    if not filename:
        return jsonify({"error": "Missing filename parameter"}), 400

    # Ensure safe path within upload folder
    safe_filename = os.path.basename(filename)
    img_path = os.path.join(UPLOAD_FOLDER, safe_filename)
    if not os.path.exists(img_path):
        return jsonify({"error": "Uploaded image not found"}), 404

    try:
        pil_img = Image.open(img_path).convert("L")
        img_arr = np.array(pil_img, dtype=np.uint8)
        H, W = img_arr.shape

        # Determine execution target
        run_on_gpu = False
        if target_device == "cuda":
            if not CUDA_AVAILABLE:
                return jsonify({"error": "CUDA GPU requested but hardware or driver is unavailable. Switch to CPU mode."}), 400
            run_on_gpu = True
        elif target_device == "cpu":
            run_on_gpu = False
        else:  # auto
            run_on_gpu = CUDA_AVAILABLE

        t_start = time.perf_counter()
        timing_breakdown = {}
        output_arr = None

        if run_on_gpu:
            if algorithm == "gaussian":
                res = cuda_gaussian_blur(img_arr, sigma=sigma, use_shared_memory=use_shared_memory)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"]}
            elif algorithm == "hist_eq":
                res = cuda_histogram_equalization(img_arr)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"]}
            elif algorithm == "sharpen":
                blur_res = cuda_gaussian_blur(img_arr, sigma=sigma, use_shared_memory=use_shared_memory)
                res = cuda_unsharp_mask(img_arr, blur_res["output"], amount=amount)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"] + blur_res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"] + blur_res["total_ms"]}
            elif algorithm == "gamma":
                res = cuda_gamma_correction(img_arr, gamma=gamma)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"]}
            elif algorithm == "sobel":
                res = cuda_sobel(img_arr)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"]}
            elif algorithm == "full_pipeline":
                res = cuda_full_pipeline(img_arr, sigma=sigma, amount=amount, use_shared_memory=use_shared_memory)
                output_arr = res["output"]
                timing_breakdown = {"h2d_ms": res["h2d_ms"], "kernel_ms": res["kernel_ms"], "d2h_ms": res["d2h_ms"], "total_ms": res["total_ms"]}
            else:
                return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400
        else:
            # CPU Execution
            if algorithm == "gaussian":
                t0 = time.perf_counter()
                output_arr = cpu_gaussian_blur(img_arr, sigma=sigma)
                t_tot = (time.perf_counter() - t0) * 1000.0
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": t_tot, "d2h_ms": 0.0, "total_ms": t_tot}
            elif algorithm == "hist_eq":
                t0 = time.perf_counter()
                output_arr, _, _ = cpu_histogram_equalization(img_arr)
                t_tot = (time.perf_counter() - t0) * 1000.0
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": t_tot, "d2h_ms": 0.0, "total_ms": t_tot}
            elif algorithm == "sharpen":
                t0 = time.perf_counter()
                blurred = cpu_gaussian_blur(img_arr, sigma=sigma)
                output_arr = cpu_unsharp_mask(img_arr, blurred, amount=amount)
                t_tot = (time.perf_counter() - t0) * 1000.0
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": t_tot, "d2h_ms": 0.0, "total_ms": t_tot}
            elif algorithm == "gamma":
                t0 = time.perf_counter()
                output_arr = cpu_gamma_correction(img_arr, gamma=gamma)
                t_tot = (time.perf_counter() - t0) * 1000.0
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": t_tot, "d2h_ms": 0.0, "total_ms": t_tot}
            elif algorithm == "sobel":
                t0 = time.perf_counter()
                output_arr = cpu_sobel_edge(img_arr)
                t_tot = (time.perf_counter() - t0) * 1000.0
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": t_tot, "d2h_ms": 0.0, "total_ms": t_tot}
            elif algorithm == "full_pipeline":
                res = cpu_full_pipeline(img_arr, sigma=sigma, amount=amount)
                output_arr = res["enhanced"]
                timing_breakdown = {"h2d_ms": 0.0, "kernel_ms": res["total_ms"], "d2h_ms": 0.0, "total_ms": res["total_ms"]}
            else:
                return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

        # Save output image
        out_filename = f"enhanced_{safe_filename}"
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        out_img = Image.fromarray(output_arr, mode="L")
        out_img.save(out_path, format="PNG")

        out_hist = cpu_histogram(output_arr).tolist()
        b64_enhanced = numpy_to_base64_png(output_arr)

        return jsonify({
            "message": "Enhancement complete",
            "output_filename": out_filename,
            "algorithm": algorithm,
            "target_used": "CUDA GPU" if run_on_gpu else "Sequential CPU",
            "device_name": CUDA_DEVICE_NAME if run_on_gpu else "Host CPU",
            "memory_tier": "Shared Memory (20x20 Tile)" if (run_on_gpu and use_shared_memory) else ("Global Memory" if run_on_gpu else "RAM"),
            "timings": timing_breakdown,
            "histogram": out_hist,
            "enhanced_image_url": b64_enhanced
        })
    except Exception as e:
        return jsonify({"error": f"Enhancement error: {str(e)}"}), 500


# ----------------------------------------------------------------------------
# 4. API: Benchmark
# ----------------------------------------------------------------------------
@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    data = request.get_json(silent=True) or {}
    resolution = int(data.get("resolution", 512))
    if resolution not in [512, 1024, 2048, 4096]:
        return jsonify({"error": "Resolution must be one of: 512, 1024, 2048, 4096"}), 400

    try:
        rec = run_single_benchmark(resolution, resolution)
        # Append or update results.csv
        records = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r["Resolution"] != f"{resolution}x{resolution}":
                        records.append(r)
        records.append(rec)
        save_results_csv(records, RESULTS_FILE)

        return jsonify({
            "message": f"Benchmark completed for {resolution}x{resolution}",
            "result": rec
        })
    except Exception as e:
        return jsonify({"error": f"Benchmark error: {str(e)}"}), 500


# ----------------------------------------------------------------------------
# 5. API: Correctness Verification
# ----------------------------------------------------------------------------
@app.route("/api/verify", methods=["POST"])
def api_verify():
    try:
        results = run_verification_suite()
        all_passed = all(r["status"] == "PASS" for r in results)
        return jsonify({
            "message": "Verification suite completed",
            "all_passed": all_passed,
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r["status"] == "PASS"),
            "environment": CUDA_DEVICE_NAME if CUDA_AVAILABLE else "CPU Verification",
            "results": results
        })
    except Exception as e:
        return jsonify({"error": f"Verification error: {str(e)}"}), 500


# ----------------------------------------------------------------------------
# 6. API: Get Existing Benchmark Results
# ----------------------------------------------------------------------------
@app.route("/api/results", methods=["GET"])
def api_results():
    if not os.path.exists(RESULTS_FILE):
        return jsonify({"records": []})

    records = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            records.append(r)
    return jsonify({"records": records})


# ----------------------------------------------------------------------------
# 7. API: Download outputs or CSV
# ----------------------------------------------------------------------------
@app.route("/api/download/<filename>", methods=["GET"])
def api_download(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name == "results.csv":
        return send_from_directory(BASE_DIR, "results.csv", as_attachment=True)
    return send_from_directory(OUTPUT_FOLDER, safe_name, as_attachment=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", cuda_available=CUDA_AVAILABLE, gpu_name=CUDA_DEVICE_NAME)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
