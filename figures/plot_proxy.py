"""Grid of validation images under every proxy style (one row per style), for the report.

Usage: python -m figures.plot_proxy [--out figures/proxy_styles.png] [--seed 0]

Columns are one validation image per class, taken from the balanced subset used by the proxy
(100 per class, --seed); rows are clean, grayscale and the five proxy styles of recvis.styles.
32 px images are upscaled with nearest neighbour so the pixels stay visible.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from recvis.data import CLASS_NAMES, NpzImages, balanced_subset, resolve_data_dir
from recvis.styles import apply_transform

ROWS = ("clean", "grayscale", "posterize", "pixelate", "sketch", "dither", "grid")
LABEL_WIDTH, HEADER = 80, 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cell", type=int, default=80)
    parser.add_argument(
        "--out", type=Path, default=Path("figures/proxy_styles.png")
    )
    args = parser.parse_args()
    try:
        args.data = resolve_data_dir(args.data)
    except FileNotFoundError as error:
        parser.error(str(error))
    return args


def main() -> None:
    args = parse_args()
    validation = NpzImages(args.data / "validation.npz", require_labels=True)
    subset = balanced_subset(validation.labels, 100, args.seed)
    labels = validation.labels[subset]
    picks = subset[[np.flatnonzero(labels == c)[0] for c in range(len(CLASS_NAMES))]]
    images = validation.images[picks]
    cell = args.cell
    grid = Image.new(
        "RGB",
        (LABEL_WIDTH + len(CLASS_NAMES) * cell, HEADER + len(ROWS) * cell),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for column, name in enumerate(CLASS_NAMES):
        draw.text((LABEL_WIDTH + column * cell + 3, 2), name, fill="black")
    for row, style in enumerate(ROWS):
        top = HEADER + row * cell
        draw.text((4, top + cell // 2 - 6), style, fill="black")
        for column, image in enumerate(apply_transform(images, style)):
            grid.paste(
                Image.fromarray(image).resize((cell, cell), Image.Resampling.NEAREST),
                (LABEL_WIDTH + column * cell, top),
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Wrote {args.out} ({grid.width} x {grid.height})")


if __name__ == "__main__":
    main()
