"""Top-level run script for the BEAST train+predict capsule.

Simplest end-to-end loop to validate BEAST (paninski-lab/beast) as an
unsupervised behavior-video autoencoder whose per-frame latents can be paired
with keypoint tracking:

    extract frames -> train a ResNet autoencoder -> predict per-frame latents

This first cut trains a fresh backbone on a single video. That is *not* how
BEAST is meant to be used at scale (train once on a representative subset, then
run cheap inference across many videos), but it proves the loop emits a
``(num_frames, num_latents)`` latent array. See README for the long-term plan.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Code Ocean conventions: data assets are mounted read-only at /data
# (symlinked to /root/capsule/data) and outputs go to /results.
DEFAULT_DATA_DIR = Path("/root/capsule/data")
DEFAULT_RESULTS_DIR = Path("/root/capsule/results")
DEFAULT_SCRATCH_DIR = Path("/root/capsule/scratch")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".mj2"}
BEAST = shutil.which("beast") or "beast"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="BEAST train+predict smoke-test capsule")
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file. Absolute, or relative to --data-dir. "
        "If omitted, the first video found under --data-dir is used.",
    )
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--scratch-dir", type=Path, default=DEFAULT_SCRATCH_DIR)
    p.add_argument("--config", type=Path, default=here / "config" / "resnet_ae.yaml")
    p.add_argument(
        "--frames-per-video",
        type=int,
        default=500,
        help="Frames extracted for training (beast extract -n).",
    )
    p.add_argument(
        "--extraction-method",
        type=str,
        default="pca_kmeans",
        help="beast extract -m (default: pca_kmeans).",
    )
    p.add_argument(
        "--num-epochs",
        type=int,
        default=2,
        help="Training epochs. Tiny by default to prove the loop; raise for real runs.",
    )
    p.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Override training batch size from config. Lower it (e.g. 128/256) "
        "if you hit a CUDA OOM; the ResNet config defaults to 512.",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Inference batch size.")
    p.add_argument(
        "--no-reconstructions",
        action="store_true",
        help="Skip saving the reconstruction MP4 (only emit latents). The MP4 "
        "writer depends on an available OpenCV codec, which can be missing.",
    )
    p.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Override num_gpus from config (0 = CPU). Default: leave config value.",
    )
    return p.parse_args()


def resolve_video(args: argparse.Namespace) -> Path:
    if args.video:
        v = Path(args.video)
        if not v.is_absolute():
            v = args.data_dir / v
        if not v.is_file():
            sys.exit(f"[run_capsule] video not found: {v}")
        return v
    candidates = sorted(
        f for f in args.data_dir.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
        sys.exit(f"[run_capsule] no videos found under {args.data_dir}")
    print(f"[run_capsule] discovered {len(candidates)} video(s); using first.")
    return candidates[0]


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n[run_capsule] === {name} ===\n[run_capsule] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    video = resolve_video(args)
    stem = video.stem

    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    # `beast extract` only scans a *directory* of videos, so isolate the target
    # video in its own input dir (via symlink) for a single-video run.
    input_dir = args.scratch_dir / stem / "input"
    frames_dir = args.scratch_dir / stem / "frames"
    model_dir = args.scratch_dir / stem / "model"
    input_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    link = input_dir / video.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(video.resolve())

    print(f"[run_capsule] video       : {video}")
    print(f"[run_capsule] frames_dir  : {frames_dir}")
    print(f"[run_capsule] model_dir   : {model_dir}")
    print(f"[run_capsule] results_dir : {args.results_dir}")

    # 1) Extract frames for training.
    run_step(
        "extract",
        [BEAST, "extract", "-i", str(input_dir), "-o", str(frames_dir),
         "-n", str(args.frames_per_video), "-m", args.extraction_method],
    )

    # 2) Train the autoencoder on the extracted frames.
    overrides = [f"training.num_epochs={args.num_epochs}"]
    if args.train_batch_size is not None:
        overrides.append(f"training.train_batch_size={args.train_batch_size}")
    train_cmd = [
        BEAST, "train",
        "-c", str(args.config),
        "-d", str(frames_dir),
        "-o", str(model_dir),
        "--overrides", *overrides,
    ]
    if args.gpus is not None:
        train_cmd += ["--gpus", str(args.gpus)]
    run_step("train", train_cmd)

    # 3) Predict per-frame latents (+ optional reconstruction) for the whole video.
    pred_dir = args.results_dir / stem
    predict_cmd = [
        BEAST, "predict", "-m", str(model_dir), "-i", str(video),
        "-o", str(pred_dir), "-b", str(args.batch_size), "--save_latents",
    ]
    if not args.no_reconstructions:
        predict_cmd.append("--save_reconstructions")
    run_step("predict", predict_cmd)

    metadata = {
        "video": str(video),
        "frames_per_video": args.frames_per_video,
        "extraction_method": args.extraction_method,
        "num_epochs": args.num_epochs,
        "config": str(args.config),
        "outputs_dir": str(pred_dir),
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }
    meta_path = args.results_dir / f"{stem}_capsule_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\n[run_capsule] wrote {meta_path}")
    print("[run_capsule] done.")


if __name__ == "__main__":
    main()
