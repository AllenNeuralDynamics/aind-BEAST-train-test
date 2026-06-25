# Plan: Analyze BEAST per-frame latents (pair with keypoints)

## Context

This repo is the **aind-BEAST-train-test** Code Ocean capsule
(github.com/AllenNeuralDynamics/aind-BEAST-train-test). It runs BEAST
(paninski-lab/beast) on a behavior video: decimate long video → `beast extract`
(pca_kmeans, 500 frames) → `beast train` (ResNet-AE, 12 latents) → `beast predict`
→ per-frame latents. Goal: an unsupervised per-frame embedding to pair with
Lightning Pose keypoint tracking.

**Done (working end-to-end on Code Ocean GPU):**
- Capsule scaffolded (`code/run_capsule.py`, `code/run`, `code/config/resnet_ae.yaml`,
  `environment/Dockerfile`, `README.md`).
- Solved the real blockers: broken `beast-backbones` wheel (install editable from a
  pinned git clone in the Dockerfile), GPU-only `accelerator` (needs a CUDA base
  image set via the CO Environment UI, not a hand-edited `FROM`), `beast extract`
  directory requirement, and the long-video OOM (decimate to `--extract-max-frames`,
  default 50000).
- First real run on `behavior_784803_2025-07-03_13-55-13/.../bottom_camera.mp4`
  (1.9M frames): GPU confirmed (`True (cuda)`), trained 100 epochs (val_loss 0.033),
  predicted over all frames → **`bottom_camera.npy`, shape (1899809, 12), ~87 MB**.

**Why now:** before scaling to a multi-session backbone, validate that these
unsupervised latents are meaningful and pair usefully with keypoints. This is the
prioritized next step.

## Immediate work: latent analysis (keep basic for this first test)

### Inputs (all attached as Code Ocean data assets under `/data`)
- **BEAST latents**: captured beast-output asset, `bottom_camera.npy` → `(1899809, 12)`.
- **Pose tracking (Lightning Pose)**: under
  `/data/keypoint_tracking_bottomview_LCrecordings_20260403/behavior_784803_2025-07-03_13-55-13/intermediate_data`
  — DLC-style LP prediction CSV(s): header row 0 = keypoint names (e.g.
  `tongue_tip_center`, `jaw`), next row(s) = `x`/`y`/`likelihood`; **one row per
  video frame**.
- **Session data**: attached for trial/timing context (optional for this first test).

### Notebook: `code/analyze_latents.ipynb` (run in a JupyterLab Cloud Workstation)
Mirror the `code/example_motion_energy.ipynb` pattern from aind-motion-energy-capsule.
Keep it linear and minimal:

1. **Load latents**: `z = np.load(.../bottom_camera.npy)`; assert `(1899809, 12)`, float32.
2. **Latent sanity check** (the key gate): per-dim mean/std — should show real spread,
   not the ~0.001 collapse from the 2-epoch smoke test; quick histograms; plot a few
   latent dims over a short frame window.
3. **Load Lightning Pose keypoints**: small inline loader —
   `pd.read_csv(csv, header=[0,1,2], index_col=0)` (DLC/LP layout) → per-keypoint
   `x, y, likelihood`, one row per frame. Reference (don't depend on):
   `load_keypoints_from_csv` + speed via `kinematics_filter` in
   github.com/AllenNeuralDynamics/aind-dynamic-foraging-behavior-video-analysis
   (`kinematics/tongue_kinematics_utils.py`).
4. **Align**: assert keypoint row count == latent row count (1,899,809); row i ↔ frame i.
   Flag/handle any mismatch (e.g. if LP ran on a different frame set).
5. **Basic latent ↔ keypoint relationship** (the core test):
   - Per-frame keypoint speed `v = sqrt(diff(x)^2 + diff(y)^2)` for a key bodypart
     (e.g. `tongue_tip_center`), mirroring `kinematics_filter`.
   - Plot latent traces vs that speed over the same window.
   - Correlation matrix: 12 latent dims vs {x, y, speed} of 1–2 keypoints.
   - (Optional) linear fit latents → a keypoint's x/y; report R².
6. **(Optional, light)**: PCA variance of the 12 dims; 2-D scatter of a ~20k-frame
   subsample colored by keypoint speed. Skip UMAP this pass (avoids a new dep).

### Environment
Capsule env already has numpy + pandas + scikit-learn (BEAST deps) — sufficient for
the basic notebook. No new packages needed (UMAP skipped).

## Roadmap beyond this (later phases, not now)
1. **Multi-session backbone** (the scientific goal): train ONE backbone on a
   representative subset of many bottom-camera sessions so latents are comparable
   across sessions; then cheap per-video inference. Needs multi-video frame
   extraction into one training set.
2. **Productionize**: split into separate `beast-train` + `beast-predict` capsules;
   add a predict-stride option (avoid the full 1.9M-frame pass when keypoints are
   lower-rate); richer training defaults.
3. **Polish**: file the upstream `beast-backbones` version-bug issue.

## Verification
- Open `code/analyze_latents.ipynb` in a JupyterLab workstation with the latents +
  keypoint assets attached; run all cells top to bottom with no errors.
- Sanity gate: per-dim std is clearly non-zero / varied (latents not collapsed);
  PCA shows the 12 dims carry structure.
- Keypoint gate: frame alignment succeeds and at least one latent↔keypoint
  relationship is quantified (correlation or regression R²).

## Files / prerequisites
- New: `code/analyze_latents.ipynb` (in the capsule repo).
- Inputs (attached): latents asset (`bottom_camera.npy`), Lightning Pose asset at the
  `intermediate_data` path above, session asset.
- To confirm when building the notebook: the exact LP CSV filename(s) under
  `intermediate_data`, the bottom-view keypoint names, and that the CSV has one row
  per full-video frame (== 1,899,809).
