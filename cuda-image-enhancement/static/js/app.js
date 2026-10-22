// Standalone Flask Frontend Client Logic
document.addEventListener("DOMContentLoaded", async () => {
  let currentFile = null;
  let targetDevice = "auto";

  // Check backend status
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    const badge = document.getElementById("device-badge");
    if (data.cuda_available) {
      badge.className = "badge bg-success px-3 py-2";
      badge.textContent = `CUDA GPU: ${data.gpu_name} (${data.cuda_capability})`;
    } else {
      badge.className = "badge bg-warning text-dark px-3 py-2";
      badge.textContent = `Mode: CPU Fallback (${data.init_message || 'No GPU'})`;
    }
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }

  // Sliders
  const gammaRange = document.getElementById("gammaRange");
  const gammaVal = document.getElementById("gammaVal");
  gammaRange?.addEventListener("input", (e) => gammaVal.textContent = e.target.value);

  const sharpRange = document.getElementById("sharpRange");
  const sharpVal = document.getElementById("sharpVal");
  sharpRange?.addEventListener("input", (e) => sharpVal.textContent = e.target.value);

  // Device buttons
  const autoBtn = document.getElementById("btn-device-auto");
  const cudaBtn = document.getElementById("btn-device-cuda");
  const cpuBtn = document.getElementById("btn-device-cpu");

  function setDevice(dev, btn) {
    targetDevice = dev;
    [autoBtn, cudaBtn, cpuBtn].forEach(b => b?.classList.remove("active", "btn-primary", "btn-info", "btn-secondary"));
    [autoBtn, cudaBtn, cpuBtn].forEach(b => b?.classList.add("btn-outline-secondary"));
    btn?.classList.remove("btn-outline-secondary");
    btn?.classList.add("active", dev === 'cuda' ? 'btn-primary' : (dev === 'auto' ? 'btn-info' : 'btn-secondary'));
  }

  autoBtn?.addEventListener("click", () => setDevice("auto", autoBtn));
  cudaBtn?.addEventListener("click", () => setDevice("cuda", cudaBtn));
  cpuBtn?.addEventListener("click", () => setDevice("cpu", cpuBtn));

  // Image Upload
  const imageInput = document.getElementById("imageInput");
  imageInput?.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const metricsBar = document.getElementById("metricsBar");
    metricsBar.textContent = "Uploading image...";

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }
      currentFile = data.filename;
      document.getElementById("originalImg").src = data.preview_url;
      metricsBar.textContent = `Uploaded: ${data.width}x${data.height} (${data.pixels.toLocaleString()} pixels)`;
    } catch (err) {
      alert("Upload failed: " + err.message);
    }
  });

  // Run Enhancement
  const runBtn = document.getElementById("runBtn");
  runBtn?.addEventListener("click", async () => {
    if (!currentFile) {
      alert("Please upload an image first.");
      return;
    }
    const algorithm = document.getElementById("algorithmSelect").value;
    const useShared = document.getElementById("sharedMemToggle").checked;
    const gamma = parseFloat(gammaRange.value);
    const amount = parseFloat(sharpRange.value);

    runBtn.disabled = true;
    runBtn.textContent = "Processing...";

    try {
      const res = await fetch("/api/enhance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: currentFile,
          algorithm,
          target_device: targetDevice,
          use_shared_memory: useShared,
          gamma,
          amount
        })
      });
      const data = await res.json();
      if (data.error) {
        alert(data.error);
        return;
      }

      document.getElementById("enhancedImg").src = data.enhanced_image_url;
      document.getElementById("timeH2D").textContent = `${data.timings.h2d_ms.toFixed(2)} ms`;
      document.getElementById("timeKernel").textContent = `${data.timings.kernel_ms.toFixed(2)} ms`;
      document.getElementById("timeD2H").textContent = `${data.timings.d2h_ms.toFixed(2)} ms`;
      document.getElementById("timeTotal").textContent = `${data.timings.total_ms.toFixed(2)} ms`;

      document.getElementById("metricsBar").textContent = `Processed with ${data.target_used} (${data.memory_tier}) in ${data.timings.total_ms.toFixed(2)} ms`;
    } catch (err) {
      alert("Enhancement failed: " + err.message);
    } finally {
      runBtn.disabled = false;
      runBtn.textContent = "Run Enhancement";
    }
  });
});
