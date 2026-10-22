# CUDA-Accelerated Image Enhancement System
**Course:** B.Tech Artificial Intelligence and Data Science  
**Subject:** Accelerated AI Development Using CUDA  
**Author:** Student Project Submission  

---

## 1. Project Abstract
Image enhancement operations such as spatial filtering, contrast adjustment, and edge detection are fundamental stages in computer vision pipelines. When executed sequentially on standard Central Processing Units (CPUs), high-resolution imagery (e.g., 4K medical imaging, satellite telemetry, or video surveillance frames) incurs severe computational bottlenecks due to the $O(H \times W \times K^2)$ complexity of 2D spatial convolution and pixel-wise remapping.

This project implements a high-performance **CUDA-Accelerated Image Enhancement System** using **NVIDIA CUDA** and **Numba CUDA** in Python. The system delivers an interactive web dashboard for real-time image processing, benchmarks sequential CPU versus massively parallel GPU execution, and evaluates memory hierarchy optimization via **Shared-Memory Tiling** (with a 20×20 shared tile and 2-pixel halo cooperative loading) compared to un-cached Global Memory access.

---

## 2. Key Architecture & Implemented Algorithms

| Algorithm | CPU Implementation | CUDA Global Memory | CUDA Shared-Memory Tiling |
| :--- | :--- | :--- | :--- |
| **5×5 Gaussian Denoise** | Sequential 2D nested loop with clamp boundary | 1 thread/pixel, reads 25 global memory addresses per pixel | 16×16 thread block cooperatively loads 20×20 tile (400 elements) into `__shared__` memory; zero redundant global reads |
| **Histogram Equalization** | Sequential 256-bin accumulation, CDF, and LUT | Shared memory histogram per block, atomic additions to global histogram | Parallel LUT remapping kernel: $O(1)$ memory access per thread |
| **Unsharp-Mask Sharpening** | $I_{sharp} = I_{orig} + \alpha(I_{orig} - I_{blur})$ | Element-wise parallel float compute with uint8 clamp | Fast register math with coalesced memory write |
| **Gamma Correction** | $255 \times (I / 255)^\gamma$ lookup table | Parallel LUT application kernel | Coalesced GPU execution |
| **Sobel Edge Detection** | $3\times 3$ $G_x, G_y$ gradient magnitude | 16×16 threads per block with clamped borders | Fast register stencil computation |
| **Full 7-Stage Pipeline** | Denoise $\to$ HistEq $\to$ Blur $\to$ Sharpen | Intermediate buffers preserved in GPU VRAM | Eliminates redundant Host-to-Device (H2D) transfers |

---

## 3. Technology Stack

- **Host Language:** Python 3.11
- **Web Server:** Flask 3.0.3, Werkzeug
- **Array & Image Engine:** NumPy 1.26.4, Pillow 10.3.0
- **GPU Accelerated Compiler:** Numba CUDA 0.59.1
- **Visual Analytics:** Matplotlib 3.8.4, Chart.js
- **Frontend Dashboard:** Modern Dark Glassmorphism, Tailwind CSS, Responsive Flexbox

---

## 4. Hardware & Software Prerequisites

1. **Operating System:** Windows 10/11 (64-bit) or Ubuntu Linux 22.04 LTS
2. **GPU:** NVIDIA GPU with Compute Capability $\ge 5.0$ (e.g., GTX 1060, RTX 2060/3060/4060, Tesla T4, A100)
3. **Driver:** NVIDIA Driver version $\ge 525.0$
4. **CUDA Toolkit:** CUDA Toolkit 11.8 or 12.x installed with `nvcc` in PATH
5. **Python:** Python 3.11.x (64-bit)

> **Note on CPU Fallback:** If executed on a machine without an NVIDIA GPU (such as standard integrated graphics laptops), the application automatically detects GPU absence and activates **CPU Fallback Mode**. All algorithms and tests execute using pure CPU reference code without crashing.

---

## 5. Step-by-Step Installation Guide (Windows PowerShell)

Open **PowerShell as Administrator** or standard user in the project directory:

### Step 5.1: Verify Python Version
```powershell
python --version
# Output must indicate Python 3.11.x
```

### Step 5.2: Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 5.3: Verify NVIDIA Driver & GPU
```powershell
nvidia-smi
```
Verify that your GPU model and CUDA Version are displayed.

### Step 5.4: Install Python Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5.5: Verify CUDA from Python
Run this quick one-liner to verify Numba sees your GPU:
```powershell
python -c "from numba import cuda; print('CUDA Available:', cuda.is_available()); print('Device:', cuda.get_current_device().name if cuda.is_available() else 'None')"
```

---

## 6. Running the System

### 6.1 Launch the Web Dashboard
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### 6.2 Run Correctness Verification Suite
Validates pixel-by-pixel mathematical equivalence between CPU ground truth and CUDA kernels on both $256\times 256$ and odd non-multiple dimensions ($50\times 77$):
```powershell
python verify_kernels.py
```
Outputs Max Absolute Difference, Mean Absolute Error, RMSE, and PASS/FAIL status.

### 6.3 Run Automated Performance Benchmark
Executes 5 runs per resolution ($512\times 512$, $1024\times 1024$, $2048\times 2048$, $4096\times 4096$), computes medians, and generates `results.csv`:
```powershell
python benchmark.py
```

### 6.4 Generate Academic Figures & Plots
```powershell
python make_figures.py
```
Outputs high-resolution publication figures into `outputs/figures/`.

---

## 7. Running on Google Colab (Free NVIDIA T4 GPU)

If your local computer does not have a physical NVIDIA GPU, you can run genuine CUDA benchmarks on Google Colab:
1. Open [Google Colab](https://colab.research.google.com/).
2. Change runtime to GPU: **Runtime $\to$ Change runtime type $\to$ T4 GPU**.
3. Upload the project files or clone the repository.
4. Execute in a cell:
```python
!nvidia-smi
!pip install -r requirements.txt
!python verify_kernels.py
!python benchmark.py
!python make_figures.py
```

---

## 8. Viva Voce Questions & Answers (College Evaluation Prep)

### Q1: What is the primary difference between CUDA Global Memory and Shared Memory?
**Answer:** Global memory is off-chip DRAM with high latency (400–800 clock cycles) and large capacity (several gigabytes). Shared memory is on-chip SRAM located directly inside each Streaming Multiprocessor (SM), with ultra-low latency (~1–5 clock cycles) and high bandwidth, but limited capacity (usually 48KB–96KB per SM). We use shared memory in our 5×5 Gaussian filter to store a 20×20 image tile, ensuring neighboring threads reuse pixel values instead of making 25 redundant global memory fetches.

### Q2: Why is the shared memory tile size 20×20 for a 16×16 output block?
**Answer:** A 5×5 convolution filter has a filter radius of $r = \lfloor 5/2 \rfloor = 2$ pixels. To compute outputs for a 16×16 thread block, boundary threads require a 2-pixel apron (halo) on top, bottom, left, and right. Thus, the required input footprint is:
$$(16 + 2 \times 2) \times (16 + 2 \times 2) = 20 \times 20 = 400 \text{ pixels}.$$
The 256 threads cooperatively load these 400 pixels before calling `cuda.syncthreads()`.

### Q3: Why is `cuda.syncthreads()` necessary in the shared-memory Gaussian filter?
**Answer:** Because threads execute asynchronously in warps. If thread A starts computing convolution before thread B finishes loading its assigned halo pixel into the shared tile, thread A will read uninitialized memory (race condition). `cuda.syncthreads()` establishes an execution barrier ensuring all 400 pixels are loaded into shared memory before any thread begins convolution.

### Q4: How is partial block handling achieved for non-multiple dimensions like 50×77?
**Answer:** Grid dimensions are computed using ceiling division:
`blocks_x = ceil(W / 16)`, `blocks_y = ceil(H / 16)`.
Inside the kernel, an explicit boundary guard checks if the thread's global index exceeds image boundaries:
```python
r, c = cuda.grid(2)
if r < H and c < W:
    # compute valid output
```
Threads outside $[0, H-1]$ or $[0, W-1]$ safely exit without out-of-bounds memory access.

### Q5: How do we prevent race conditions in parallel histogram computation?
**Answer:** Multiple threads processing pixels with identical intensity levels would collide when updating the histogram bin. We allocate a 256-bin integer array in shared memory per block and use `cuda.atomic.add(s_hist, pixel_value, 1)`. Atomic additions serialize concurrent updates at the hardware level, eliminating read-modify-write collisions. Afterwards, block histograms are aggregated into the global histogram using atomic adds.

### Q6: What is the formula used for Speedup calculation?
**Answer:**
$$\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{CUDA}}}$$
Where $T_{\text{CPU}}$ is the sequential CPU execution time, and $T_{\text{CUDA}}$ is the synchronized CUDA execution time (reported both as pure kernel execution time and end-to-end time including data transfers).
