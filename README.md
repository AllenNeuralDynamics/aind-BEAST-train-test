# aind-BEAST-train-test

A Code Ocean capsule that runs [BEAST](https://github.com/paninski-lab/beast)
(Behavioral analysis via Self-supervised pretraining of Transformers) on behavior
videos as an **unsupervised** representation to pair with keypoint tracking.

This is the **simplest possible** first cut: it runs the full BEAST loop
end-to-end on a single video to prove it emits a per-frame latent array.

```
extract frames  ->  train ResNet autoencoder  ->  predict per-frame latents
   (beast extract)        (beast train)              (beast predict)
```

## Requirements

- **A GPU machine type on Code Ocean.** BEAST hardcodes `accelerator='gpu'` in
  both training (`beast/train.py`) and inference (`beast/inference.py`), so this
  capsule will not run on a CPU-only environment. Do **not** pass `--gpus 0`
  (BEAST divides by `num_gpus`, which raises `ZeroDivisionError`).
- BEAST is installed editable from source in the Dockerfile on purpose; the
  published `beast-backbones` wheel crashes on a version lookup. See the
  Dockerfile comment.

## Inputs

- A behavior-video data asset mounted at `/data` (Code Ocean). Supported
  extensions: `.mp4 .avi .mkv .mov .mj2`.
- `run_capsule.py` auto-discovers the first video under `/data`, or you can pass
  `--video <path>` (relative to `/data`, or absolute).

## Outputs (`/results/<video_stem>/`)

- `<stem>.npy` — **per-frame latents**, shape `(num_frames, num_latents)`. With
  the default ResNet config, `num_latents = 12`. This is the array you align with
  keypoints.
- `<stem>_reconstruction.mp4` — 224×224 autoencoder reconstruction (QC).
- `prediction_metadata.yaml` — fps, frame count, paths (written by BEAST).
- `/results/<stem>_capsule_metadata.json` — run parameters.

## Key parameters (`code/run_capsule.py`)

| Flag | Default | Notes |
|------|---------|-------|
| `--video` | auto-discover | video under `/data` |
| `--frames-per-video` | 500 | frames used for training (`beast extract -n`) |
| `--extract-max-frames` | 50000 | decimate long videos to ~this many frames before extract (avoids an OOM in BEAST's frame selection); 0 = full video |
| `--num-epochs` | 2 | **tiny smoke-test default; raise for real runs** |
| `--train-batch-size` | config (512) | lower (128/256) if you hit a CUDA OOM |
| `--batch-size` | 32 | inference batch size |
| `--gpus` | config (1) | number of GPUs (do not set 0; see note) |
| `--no-reconstructions` | off | skip the reconstruction MP4, emit latents only |

The model config lives in `code/config/resnet_ae.yaml` (copied from BEAST).
`data_dir` and `num_epochs` are overridden at runtime.

## Scope / roadmap

This first version trains a fresh backbone **per video**, which is fine for a
smoke test but is **not** how BEAST is meant to scale. The intended long-term
workflow is:

1. **Pretrain once** on a representative subset (~50–100 videos / many diverse
   frames) → a versioned backbone checkpoint. Heavy, GPU, infrequent.
2. **Infer per-video** with the frozen backbone across hundreds/thousands of
   videos. Cheap, parallel. Produces comparable latents across sessions.
3. **Re-pretrain** only on domain shift (new rig/camera/species/protocol).

Because a single shared backbone gives a common embedding space, per-frame
latents are comparable across sessions — which is what makes them useful
alongside keypoints. Once this loop is validated we can split this into a
`beast-train` capsule (rare) and a `beast-predict` capsule (per-video).
