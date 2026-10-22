/**
 * Programmatic high-fidelity sample image generator for CUDA Image Lab
 * Generates test images with realistic noise, gradients, structures, and edges.
 */

export interface SampleImageDef {
  id: string;
  name: string;
  category: string;
  description: string;
  width: number;
  height: number;
  generate: () => ImageData;
}

export const SAMPLE_IMAGES: SampleImageDef[] = [
  {
    id: 'brain_mri',
    name: 'Brain MRI Scan',
    category: 'Biomedical Imaging',
    description: 'High-contrast axial cranial tomography with subtle tissue gradients and noise.',
    width: 512,
    height: 512,
    generate: () => {
      const width = 512;
      const height = 512;
      const imgData = new ImageData(width, height);
      const data = imgData.data;
      const cx = width / 2;
      const cy = height / 2;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          const dx = (x - cx) / 210;
          const dy = (y - cy) / 235;
          const dist = Math.sqrt(dx * dx + dy * dy);

          let intensity = 10;
          // Skull outline
          if (dist > 0.88 && dist < 0.95) {
            intensity = 220 + Math.sin(x * 0.2) * 20;
          } else if (dist <= 0.88) {
            // Brain matter folds
            const cortex = Math.sin(x * 0.08) * Math.cos(y * 0.08) * 35;
            const sulci = Math.sin((x + y) * 0.05) * Math.sin((x - y) * 0.05) * 40;
            const ventricle = (Math.abs(x - cx) < 25 && Math.abs(y - cy) < 60) ? 25 : 0;
            intensity = Math.max(30, 140 + cortex + sulci - ventricle);
          }

          // Add sensor noise (simulating MRI thermal RF noise)
          const noise = (Math.random() - 0.5) * 26;
          const finalVal = Math.min(Math.max(Math.round(intensity + noise), 0), 255);

          data[idx] = finalVal;
          data[idx + 1] = finalVal;
          data[idx + 2] = finalVal;
          data[idx + 3] = 255;
        }
      }
      return imgData;
    }
  },
  {
    id: 'low_light_city',
    name: 'Low-Light Urban Nightscape',
    category: 'Low-Light Surveillance',
    description: 'Severely underexposed street view with low dynamic range and shadows.',
    width: 512,
    height: 512,
    generate: () => {
      const width = 512;
      const height = 512;
      const imgData = new ImageData(width, height);
      const data = imgData.data;

      for (let y = 0; y < height; y++) {
        const horizon = y / height;
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          // Dark underexposed gradient base
          let intensity = 15 + horizon * 45;

          // Building silhouettes
          const bldg = Math.sin(x * 0.04) * 80 + Math.cos(x * 0.015) * 50;
          if (y > 320 - bldg) {
            intensity += 30;
            // Scattered illuminated windows
            if ((x % 16 > 10) && (y % 16 > 8) && (Math.sin(x * 3 + y * 7) > 0.4)) {
              intensity = 210 + Math.random() * 45;
            }
          }

          // Street lamp glare
          const lampDist = Math.hypot(x - 380, y - 420);
          if (lampDist < 80) {
            intensity += (80 - lampDist) * 2.2;
          }

          const noise = (Math.random() - 0.5) * 18;
          const finalVal = Math.min(Math.max(Math.round(intensity + noise), 0), 255);

          data[idx] = finalVal;
          data[idx + 1] = finalVal;
          data[idx + 2] = finalVal;
          data[idx + 3] = 255;
        }
      }
      return imgData;
    }
  },
  {
    id: 'microscopic_cells',
    name: 'Cellular Micrograph',
    category: 'Microscopy',
    description: 'Biological tissue sample featuring circular membranes, granular cytoplasm, and high noise.',
    width: 512,
    height: 512,
    generate: () => {
      const width = 512;
      const height = 512;
      const imgData = new ImageData(width, height);
      const data = imgData.data;

      // Seeded centers for cells
      const cells = [
        { x: 120, y: 140, r: 55 },
        { x: 280, y: 160, r: 70 },
        { x: 410, y: 130, r: 50 },
        { x: 160, y: 320, r: 65 },
        { x: 340, y: 340, r: 80 },
        { x: 230, y: 450, r: 45 },
      ];

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          let intensity = 190; // brightfield background

          for (const cell of cells) {
            const dist = Math.hypot(x - cell.x, y - cell.y);
            if (dist < cell.r) {
              // Cytoplasm
              intensity = 110 + (dist / cell.r) * 30;
              // Nucleus
              if (dist < cell.r * 0.35) {
                intensity = 45 + Math.random() * 20;
              }
              // Membrane edge
              if (dist > cell.r - 4) {
                intensity = 35;
              }
            }
          }

          // Specimen noise
          const noise = (Math.random() - 0.5) * 35;
          const finalVal = Math.min(Math.max(Math.round(intensity + noise), 0), 255);

          data[idx] = finalVal;
          data[idx + 1] = finalVal;
          data[idx + 2] = finalVal;
          data[idx + 3] = 255;
        }
      }
      return imgData;
    }
  },
  {
    id: 'deep_space',
    name: 'Deep Space Nebula',
    category: 'Astrophotography',
    description: 'Astronomical sensor capture with high noise, faint gas structures, and pinpoint stars.',
    width: 512,
    height: 512,
    generate: () => {
      const width = 512;
      const height = 512;
      const imgData = new ImageData(width, height);
      const data = imgData.data;

      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const idx = (y * width + x) * 4;
          // Perlin-like cloud approximation
          const s1 = Math.sin(x * 0.015) * Math.cos(y * 0.015);
          const s2 = Math.sin(x * 0.03 + y * 0.02);
          let intensity = 15 + Math.max(0, s1 + s2 * 0.5) * 75;

          // Stars
          if (Math.random() < 0.003) {
            intensity += 160 + Math.random() * 95;
          }

          const noise = (Math.random() - 0.5) * 22;
          const finalVal = Math.min(Math.max(Math.round(intensity + noise), 0), 255);

          data[idx] = finalVal;
          data[idx + 1] = finalVal;
          data[idx + 2] = finalVal;
          data[idx + 3] = 255;
        }
      }
      return imgData;
    }
  }
];
