"""Encode every image set once with the frozen encoders (GPU recommended), into an embedding cache.

Usage: python encode.py [--models pe-l14 pe-g14-224] [--data DATA_DIR] [--cache cache/features]

For each encoder, writes <cache>/<alias>/<key>.npy (L2-normalized float16 embeddings):
  val-<style>-n100-s0      the 1,000-image balanced validation subset (seed 0), clean, grayscale and the
                           five proxy styles of recvis/styles.py
  test-none                the 51,077 test images, each encoded on its own
  train-clean-n1000-s0     10,000 CIFAR-10 training images (1,000 per class, seed 0), for the CIFAR head
and the 80 prompt embeddings of every class (texts.npy). The generated images are encoded by
generate_synthetic.py. On a remote GPU run by gpurun, the cache goes to $GPURUN_OUTPUT_DIR.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from recvis.backbones import BACKBONES, FeatureStore, default_device, free_device_memory
from recvis.data import NpzImages, balanced_subset, resolve_data_dir
from recvis.heads import prototypes
from recvis.styles import VAL_STYLES, apply_transform

REMOTE_OUTPUT = os.environ.get("GPURUN_OUTPUT_DIR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=["pe-l14", "pe-g14-224"], choices=sorted(BACKBONES))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(REMOTE_OUTPUT) / "cache" / "features" if REMOTE_OUTPUT else Path("cache/features"),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--no-train", action="store_true", help="Skip the CIFAR-10 training subset.")
    args = parser.parse_args()
    try:
        args.data = resolve_data_dir(args.data)
    except FileNotFoundError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    device = default_device()
    validation = NpzImages(args.data / "validation.npz", require_labels=True)
    val_images = validation.images[balanced_subset(validation.labels, 100, 0)]
    test_images = NpzImages(args.data / "test.npz").images
    sets = {f"val-{style}-n100-s0": (lambda s=style: apply_transform(val_images, s)) for style in VAL_STYLES}
    sets["test-none"] = lambda: test_images
    if not args.no_train:
        train = NpzImages(args.data / "train.npz", require_labels=True)
        sets["train-clean-n1000-s0"] = lambda: train.images[balanced_subset(train.labels, 1000, 0)]
    print(f"device {device}, cache {args.cache}", flush=True)
    for alias in args.models:
        start = time.perf_counter()
        store = FeatureStore(BACKBONES[alias], args.cache, device, args.batch_size)
        for key, make in sets.items():
            store.images(key, make)
        prototypes(store)  # encodes and caches the prompt embeddings
        print(f"=== {alias} in {time.perf_counter() - start:.0f}s", flush=True)
        if "backbone" in store.__dict__:
            del store.__dict__["backbone"]
        free_device_memory(device)


if __name__ == "__main__":
    main()
