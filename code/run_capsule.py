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
        "--extract-max-frames",
        type=int,
        default=50000,
        help="Cap the number of frames `beast extract` scans. Longer videos are "
        "temporally decimated to roughly this many frames first, because BEAST "
        "loads every downsampled frame into RAM and OOMs on long videos. "
        "0 disables decimation (use the full video).",
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


def count_frames(video: Path) -> int:
    """Best-effort frame count via ffprobe (nb_frames, else duration * fps)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames,avg_frame_rate,duration",
             "-of", "default=noprint_wrappers=1:nokey=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
        fields = dict(
            line.split("=", 1) for line in out.splitlines() if "=" in line
        )
        nb = fields.get("nb_frames", "N/A")
        if nb.isdigit() and int(nb) > 0:
            return int(nb)
        num, _, den = fields.get("avg_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den) if den and float(den) else 0.0
        return int(fps * float(fields.get("duration", "0") or 0))
    except (subprocess.CalledProcessError, ValueError, ZeroDivisionError):
        return 0


def count_keyframes(video: Path) -> int:
    """Count video keyframes (I-frames) via ffprobe packet flags (demux only)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=flags", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout
        return sum(1 for line in out.splitlines() if line.startswith("K"))
    except subprocess.CalledProcessError:
        return 0


def prepare_extract_input(
    video: Path, input_dir: Path, max_frames: int, min_candidates: int,
) -> None:
    """Stage the video for `beast extract`.

    Short videos are symlinked. Long videos are decimated to a bounded candidate
    pool first: BEAST's pca_kmeans loads every (downsampled) frame into RAM, so a
    multi-million-frame video OOM-kills the extract step. Decimation only affects
    which frames are *candidates* for the training set; inference still runs on
    the full-resolution original video.

    Two decimation strategies:
    - keyframe-only decode (`-skip_frame nokey`): fast (inter-frames are never
      decoded), used when the video has a usable number of keyframes. This is the
      common case for H.264/GOP video.
    - every-Nth full decode (`select`): fallback for all-intra codecs (e.g.
      `.mj2`, where keyframes ~= all frames) and sparse-/short-GOP video.
    """
    total = count_frames(video) if max_frames > 0 else 0
    decimate = max_frames > 0 and total > max_frames
    dest = input_dir / (video.stem + (".mp4" if decimate else video.suffix))
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    if not decimate:
        if total:
            print(f"[run_capsule] extract on full video ({total} frames)")
        dest.symlink_to(video.resolve())
        return

    n_keyframes = count_keyframes(video)
    if min_candidates <= n_keyframes <= max_frames:
        print(f"[run_capsule] decimating {total} -> {n_keyframes} keyframes "
              f"(I-frames only) for extraction")
        run_step(
            "decimate-for-extract",
            ["ffmpeg", "-y", "-skip_frame", "nokey", "-i", str(video),
             "-an", "-vsync", "vfr", str(dest)],
        )
        return

    step = -(-total // max_frames)  # ceil
    print(f"[run_capsule] decimating {total} -> ~{total // step} frames "
          f"(every {step}th; {n_keyframes} keyframes unsuitable) for extraction")
    run_step(
        "decimate-for-extract",
        ["ffmpeg", "-y", "-i", str(video), "-vf", f"select=not(mod(n\\,{step}))",
         "-vsync", "vfr", "-an", str(dest)],
    )


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

    print(f"[run_capsule] video       : {video}")
    print(f"[run_capsule] frames_dir  : {frames_dir}")
    print(f"[run_capsule] model_dir   : {model_dir}")
    print(f"[run_capsule] results_dir : {args.results_dir}")

    # `beast extract` only scans a *directory* of videos, so isolate the target
    # video in its own input dir; long videos are decimated to bound RAM.
    prepare_extract_input(
        video, input_dir, args.extract_max_frames, args.frames_per_video,
    )

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
