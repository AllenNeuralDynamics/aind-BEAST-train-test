# Plan: BEAST latents ↔ keypoints

**Capsule:** aind-BEAST-train-test. Pipeline: decimate video → `beast extract`
→ `beast train` (ResNet-AE, 12 latents) → `beast predict` → per-frame latents.
Goal: an unsupervised per-frame embedding to pair with Lightning Pose keypoints.

## Status

- ✅ Capsule works end-to-end on Code Ocean GPU. First real run on
  `bottom_camera.mp4` (1.9M frames): 100 epochs, val_loss 0.033 →
  **`bottom_camera.npy`, shape (1899809, 12)**.
- ✅ **Latents validated** in `code/analyze_latents.ipynb` (run on a JupyterLab
  workstation). Key findings:
  - **No collapse:** all 12 dims have real spread (std ~1.8–6.5).
  - **Encode pose:** latents linearly decode tongue position
    (`tongue_tip_center` y R²=0.93, x R²=0.73); speed poorly (R²=0.07) — expected
    for a static per-frame AE. Latent *velocity* recovers movement (R²~0.21,
    higher with smoothing).
  - **Movement-locked:** peri-onset averages aligned to `tongue_movs` onsets show
    latent speed `||Δz||` peaking exactly at onset, with a dim-selective subspace
    (z0/z3 +, z6/z8/z10 −) and rhythmic side-bands at the lick interval.
  - **Time-base alignment solved:** session_time→behavior_time→frame via the
    upstream `video_alignment` module + per-frame `Behav_Time` from
    `bottom_camera.csv` (exact, no fps assumption). Keypoint-speed peak at frame 0
    confirms it.
- **Conclusion:** latents are behaviorally meaningful → green light for the
  multi-session backbone.

## Notebook: `code/analyze_latents.ipynb`

Explicit asset paths (no globbing). Sections: load latents → latent gate
(spread/histograms/stacked traces) → load LP CSV → align → keypoint gate
(speed, correlation, R²) → PCA → §7 peek processed parquet/NWB →
§8 dynamics + event-aligned (latent velocity, smoothing sweep, peri-onset
figures with `video_alignment` time-base mapping).

## Next step: held-out data validation

Validate the latents generalize beyond the training session — e.g. apply the
trained backbone to a held-out video/session and check the same pose-decoding and
movement-locking hold. **(To flesh out later.)**

## Roadmap (later)

1. **Multi-session backbone** — train one backbone on a subset of many sessions
   so latents are comparable; then cheap per-video inference.
2. **Productionize** — split `beast-train` + `beast-predict`; add predict-stride.
3. File the upstream `beast-backbones` version-bug issue.
