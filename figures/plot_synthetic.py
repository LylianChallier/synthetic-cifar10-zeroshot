"""Grid of generated images (one row per class), for the report.

Usage: python -m figures.plot_synthetic --dir output/synthetic/<name> [--per-class 8] [--source 32|full]
                                         [--out figures/synthetic_<name>_<source>.png]

Reads the folder written by generate_synthetic.py. --source 32 shows what the classifiers
see (32 px, upscaled with nearest neighbour so the pixels stay visible); --source full shows the
generated images. Columns are a random sample of each class (--seed), with the template index under
each image, so the figure can be matched with metadata.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from recvis.data import CLASS_NAMES

LABEL_WIDTH, CAPTION = 90, 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--source", choices=("32", "full"), default="32")
    parser.add_argument("--cell", type=int, default=96, help="Cell size in pixels.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--templates",
        nargs="+",
        type=int,
        default=None,
        help="Template indices, one per column (default: random images).",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    for name in ("images32.npz", "metadata.csv"):
        if not (args.dir / name).exists():
            parser.error(
                f"{args.dir} has no {name}: run generate_synthetic.py."
            )
    if args.per_class < 1 or args.cell < 16:
        parser.error("--per-class must be positive and --cell at least 16.")
    args.out = args.out or Path(
        f"figures/synthetic_{args.dir.name}_{args.source}.png"
    )
    return args


def main() -> None:
    args = parse_args()
    meta = pd.read_csv(args.dir / "metadata.csv")
    with np.load(args.dir / "images32.npz") as saved:
        small = saved["images"]
    if len(small) != len(meta):
        raise SystemExit(f"{len(small)} images but {len(meta)} metadata rows.")
    rng = np.random.default_rng(args.seed)
    cell, row_height = args.cell, args.cell + CAPTION
    grid = Image.new(
        "RGB",
        (LABEL_WIDTH + len(args.templates or range(args.per_class)) * cell, len(CLASS_NAMES) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for label, name in enumerate(CLASS_NAMES):
        rows = np.flatnonzero(meta["label"].to_numpy() == label)
        if len(rows) == 0:
            raise SystemExit(f"No image of class {name}.")
        if args.templates is None:
            chosen = sorted(
                rng.choice(rows, size=min(args.per_class, len(rows)), replace=False)
            )
        else:
            templates = meta["template"].to_numpy()
            chosen = [rng.choice(rows[templates[rows] == t]) for t in args.templates]
        top = label * row_height
        draw.text((6, top + cell // 2 - 6), name, fill="black")
        for column, row in enumerate(chosen):
            if args.source == "32":
                image = Image.fromarray(small[row]).resize(
                    (cell, cell), Image.Resampling.NEAREST
                )
            else:
                image = Image.open(args.dir / meta["file"].iloc[row]).convert("RGB")
                image = image.resize((cell, cell), Image.Resampling.BICUBIC)
            left = LABEL_WIDTH + column * cell
            grid.paste(image, (left, top))
            draw.text(
                (left + 3, top + cell + 1),
                f"t{meta['template'].iloc[row]}",
                fill="gray",
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
    print(f"Wrote {args.out} ({grid.width} x {grid.height})")


if __name__ == "__main__":
    main()
