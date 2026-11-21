"""Generate a labelled synthetic training set with an open text-to-image model, then encode it.

Usage: python generate_synthetic.py --name NAME [--per-class 1000] [--encode pe-l14 pe-g14-224]

Hypothesis (after E014): a linear head trained on CIFAR-10 photos learns CIFAR's content and hurts on
the test (web and diffusion images); a head trained on generated images of the ten classes, rendered
in many ways, should not (He et al., ICLR 2023, arXiv 2210.07574).
Prompts: the 80 ImageNet templates released with CLIP (Radford et al. 2021) filled with each class
name (recvis/prompts.py). No prompt is written for this project.
Image i of a class uses template i mod (number of templates) and seed --seed + its id, so the set is
reproducible and every template is used equally. Classes are interleaved: a run stopped by
--time-budget or by an error still yields a balanced set (truncated to whole rounds over the classes).
Generator: SDXL-Turbo (adversarial diffusion distillation, Sauer et al., arXiv 2311.17042), one step,
no guidance, 512 px. Not Stable Diffusion 2.1: CIFAR-10-W, the source of the test set, was generated
with it and is excluded from the challenge.
Writes <output>/synthetic/<name>/ (<output> is $GPURUN_OUTPUT_DIR on a remote GPU, else output/):
  images32.npz           images (N, 32, 32, 3) uint8, labels, ids: the format of train.npz
  full/<class>/<id>.jpg  the generated images at full resolution (figures, reusable dataset)
  metadata.csv           id, label, class, template, prompt, seed, file
  config.json            generator, settings, counts, timings, GPU and library versions
and, with --encode, the embeddings of images32 in <cache>/<alias>/synth-<name>.npy (predict.py).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from recvis.backbones import BACKBONES, FeatureStore, default_device, free_device_memory
from recvis.data import CLASS_NAMES
from recvis.prompts import class_prompts

REMOTE_OUTPUT = os.environ.get("GPURUN_OUTPUT_DIR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--name", required=True, help="Folder and cache key, e.g. sdxlt-p80."
    )
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--model", default="stabilityai/sdxl-turbo")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--time-budget", type=float, default=360, help="Minutes of generation at most."
    )
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--encode", nargs="*", default=[], choices=sorted(BACKBONES))
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(REMOTE_OUTPUT) if REMOTE_OUTPUT else Path("output"),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(REMOTE_OUTPUT) / "cache" / "features"
        if REMOTE_OUTPUT
        else Path("cache/features"),
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        parser.error("--name may only hold letters, digits, '_', '.' and '-'.")
    if args.per_class < 1 or args.steps < 1 or args.batch_size < 1:
        parser.error("--per-class, --steps and --batch-size must be positive.")
    if args.size % 8 or not 1 <= args.jpeg_quality <= 100:
        parser.error("--size must be a multiple of 8 and --jpeg-quality in [1, 100].")
    args.prompts = class_prompts()
    return args


def build_jobs(prompts: list[list[str]], per_class: int, seed: int) -> pd.DataFrame:
    """One row per image; image i of every class comes before image i + 1 of any class."""
    rows = []
    for i in range(per_class):
        for label, options in enumerate(prompts):
            index, template = len(rows), i % len(options)
            rows.append(
                {
                    "id": index,
                    "label": label,
                    "class": CLASS_NAMES[label],
                    "template": template,
                    "prompt": options[template],
                    "seed": seed + index,
                    "file": f"full/{CLASS_NAMES[label]}/{index:05d}.jpg",
                }
            )
    return pd.DataFrame(rows)


def load_pipeline(model: str, device: torch.device):
    from diffusers import AutoPipelineForText2Image

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        options = {"variant": "fp16"} if dtype == torch.float16 else {}
        pipe = AutoPipelineForText2Image.from_pretrained(
            model, torch_dtype=dtype, **options
        )
    except (OSError, ValueError):  # no fp16 weights published for this model
        pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_slicing()  # decode one image at a time: lower peak memory
    return pipe


def generate(
    pipe,
    jobs: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    small: np.ndarray,
    progress: dict[str, int],
) -> None:
    """Generate the jobs in order into out_dir and small, until done or out of time.

    progress["done"] counts the images saved so far, so the caller can keep them after an error.
    """
    generator_device = "cpu" if device.type == "mps" else device.type
    start, next_log = time.perf_counter(), 0.0
    for begin in range(0, len(jobs), args.batch_size):
        batch = jobs.iloc[begin : begin + args.batch_size]
        generators = [
            torch.Generator(generator_device).manual_seed(int(seed))
            for seed in batch["seed"]
        ]
        with torch.inference_mode():
            images = pipe(
                prompt=batch["prompt"].tolist(),
                num_inference_steps=args.steps,
                guidance_scale=0.0,
                height=args.size,
                width=args.size,
                generator=generators,
            ).images
        for row, image in zip(batch.itertuples(), images, strict=True):
            image = image.convert("RGB")
            image.save(out_dir / row.file, quality=args.jpeg_quality)
            small[row.id] = np.asarray(image.resize((32, 32), Image.Resampling.BICUBIC))
            progress["done"] = row.id + 1
        elapsed, done = time.perf_counter() - start, progress["done"] / len(jobs)
        if begin == 0 or done >= next_log:
            print(
                f"  {progress['done']}/{len(jobs)} images, {progress['done'] / elapsed:.2f} img/s, "
                f"{elapsed / done * (1 - done) / 60:.0f} min left",
                flush=True,
            )
            next_log = done + 0.05
        if elapsed > args.time_budget * 60:
            print(f"  time budget of {args.time_budget:g} min reached", flush=True)
            return


def save_set(
    out_dir: Path, jobs: pd.DataFrame, small: np.ndarray, done: int
) -> pd.DataFrame:
    """Keep whole rounds over the classes (a balanced set), drop the rest, write npz and csv."""
    kept = done - done % len(CLASS_NAMES)
    for file in jobs["file"].iloc[kept:done]:
        (out_dir / file).unlink(missing_ok=True)
    meta = jobs.iloc[:kept]
    np.savez_compressed(
        out_dir / "images32.npz",
        images=small[:kept],
        labels=meta["label"].to_numpy(np.int64),
        ids=meta["id"].to_numpy(np.int64),
    )
    meta.to_csv(out_dir / "metadata.csv", index=False)
    return meta


def main() -> None:
    args = parse_args()
    device = default_device()
    out_dir = args.output / "synthetic" / args.name
    if (out_dir / "images32.npz").exists():
        raise SystemExit(f"{out_dir} already holds a set: pick another --name.")
    for name in CLASS_NAMES:
        (out_dir / "full" / name).mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args.prompts, args.per_class, args.seed)
    print(
        f"{len(jobs)} images ({len(args.prompts[0])} prompts per class) with {args.model} "
        f"on {device}, into {out_dir}",
        flush=True,
    )

    start = time.perf_counter()
    pipe = load_pipeline(args.model, device)
    print(f"  pipeline loaded in {time.perf_counter() - start:.0f}s", flush=True)
    small = np.zeros((len(jobs), 32, 32, 3), np.uint8)
    progress = {"done": 0}
    start = time.perf_counter()
    try:
        generate(pipe, jobs, args, device, out_dir, small, progress)
    except Exception as error:  # keep what was generated before the failure
        finish(args, device, out_dir, jobs, small, progress["done"], start, error)
        raise
    meta = finish(args, device, out_dir, jobs, small, progress["done"], start, None)
    if meta.empty:
        raise SystemExit("No complete round over the classes: nothing to encode.")

    del pipe
    free_device_memory(device)
    images = small[: len(meta)]
    for alias in args.encode:
        begin = time.perf_counter()
        store = FeatureStore(
            BACKBONES[alias], args.cache, device, args.encode_batch_size
        )
        features = store.images(f"synth-{args.name}", lambda: images)
        if len(features) != len(images):
            raise SystemExit(f"{alias}: stale cache {store.dir}/synth-{args.name}.npy.")
        print(
            f"=== {alias}: {features.shape} in {time.perf_counter() - begin:.0f}s",
            flush=True,
        )
        if "backbone" in store.__dict__:
            del store.__dict__["backbone"]
        free_device_memory(device)


def finish(
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
    jobs: pd.DataFrame,
    small: np.ndarray,
    done: int,
    start: float,
    error: Exception | None,
) -> pd.DataFrame:
    """Save the balanced part of what was generated, and the settings and timings in config.json."""
    import diffusers

    seconds = time.perf_counter() - start
    meta = save_set(out_dir, jobs, small, done)
    config = {
        "model": args.model,
        "steps": args.steps,
        "guidance_scale": 0.0,
        "size": args.size,
        "downsampling": "PIL bicubic (antialiased) to 32 x 32",
        "prompt_source": "CLIP ImageNet templates (80)",
        "prompts_per_class": len(args.prompts[0]),
        "per_class_requested": args.per_class,
        "per_class_kept": len(meta) // len(CLASS_NAMES),
        "seed": args.seed,
        "generation_seconds": round(seconds),
        "images_per_second": round(done / seconds, 3) if seconds else None,
        "device": torch.cuda.get_device_name(0)
        if device.type == "cuda"
        else device.type,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "python": platform.python_version(),
        "error": None if error is None else repr(error),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"=== kept {len(meta)} images in {seconds / 60:.0f} min", flush=True)
    return meta


if __name__ == "__main__":
    main()
