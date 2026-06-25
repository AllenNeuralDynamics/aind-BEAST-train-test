# Plan: BEAST latents ↔ keypoints

**Capsule:** aind-BEAST-train-test. Pipeline: decimate video → `beast extract`
→ `beast train` (ResNet-AE, 12 latents) → `beast predict` → per-frame latents.
Goal: an unsupervised per-frame embedding to pair with Lightning Pose keypoints.

## Status

- ✅ Capsule works end-to-end on Code Ocean GPU. First real run on
  `bottom_camera.mp4` (1.9M frames): 100 epochs, val_loss 0.033 →
  **`bottom_camera.npy`, shape (1899809, 12)**.
- ✅ Analysis notebook written: `code/analyze_latents.ipynb` (linear, minimal;
  glob-discovers assets under `/data`). **Next: run it in a JupyterLab
  workstation with the latents + keypoint assets attached.**

## Notebook: `code/analyze_latents.ipynb`

Validates the latents before scaling to a multi-session backbone. Steps:
1. Load latents (`bottom_camera.npy`), assert shape/dtype.
2. **Latent gate:** per-dim mean/std (real spread, not the smoke-test collapse),
   histograms, short-window traces.
3. Load Lightning Pose CSV (DLC layout, one row/frame) → per-keypoint x/y/likelihood.
4. Align: keypoint rows == latent rows; flag mismatch.
5. **Keypoint gate:** keypoint speed `sqrt(dx²+dy²)`; latent↔{x,y,speed}
   correlation + linear-fit R².
6. (Optional) PCA variance; 2-D scatter colored by speed.

**To confirm on first run:** exact LP CSV filename, bottom-view keypoint names,
one row per full-video frame (== 1,899,809). Env already has numpy/pandas/sklearn.

## Roadmap (later)

1. **Multi-session backbone** — train one backbone on a subset of many sessions
   so latents are comparable; then cheap per-video inference.
2. **Productionize** — split `beast-train` + `beast-predict`; add predict-stride.
3. File the upstream `beast-backbones` version-bug issue.
