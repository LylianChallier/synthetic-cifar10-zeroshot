"""Styles of the validation proxy (posterize, pixelate, sketch, dither, grid) and grayscale.

Each transform maps one 32 x 32 RGB image to another; they mimic low-level renderings seen in the test.
"""

from __future__ import annotations

import zlib
from collections.abc import Callable

import numpy as np
from PIL import Image, ImageFilter

ImageFn = Callable[[Image.Image], Image.Image]


def _posterize(image: Image.Image, colors: int = 6) -> Image.Image:
    return image.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).convert("RGB")


def _dither(image: Image.Image, ink: int = 30, paper: int = 170) -> Image.Image:
    dots = np.asarray(
        image.convert("L").convert("1"), dtype=bool
    )  # Floyd-Steinberg dithering
    gray = np.where(dots, paper, ink).astype(np.uint8)
    return Image.fromarray(np.stack([gray] * 3, axis=-1))


def _pixelate(image: Image.Image, factor: int = 2) -> Image.Image:
    small = image.resize((image.width // factor, image.height // factor), Image.BOX)
    return small.resize(image.size, Image.NEAREST)


def _sketch(image: Image.Image) -> Image.Image:
    return image.convert("L").filter(ImageFilter.CONTOUR).convert("RGB")


def _grid_overlay(image: Image.Image) -> Image.Image:
    """Draw an irregular white lattice, seeded by the image content so the proxy is reproducible."""
    rgb = np.asarray(image).copy()
    rng = np.random.default_rng(zlib.crc32(rgb.tobytes()))
    color = int(rng.integers(225, 256))
    directions = (
        ("v", "h") if rng.random() < 0.8 else (("v",) if rng.random() < 0.5 else ("h",))
    )
    for direction in directions:
        position = int(rng.integers(0, 6))
        while position < rgb.shape[1]:
            width = int(rng.integers(1, 3))
            if direction == "v":
                rgb[:, position : position + width] = color
            else:
                rgb[position : position + width, :] = color
            position += width + int(rng.integers(3, 9))
    return Image.fromarray(rgb)


TRANSFORMS: dict[str, ImageFn] = {
    "clean": lambda image: image,
    "grayscale": lambda image: image.convert("L").convert("RGB"),
    "posterize": _posterize,
    "pixelate": _pixelate,
    "sketch": _sketch,
    "dither": _dither,
    "grid": _grid_overlay,
}
VAL_STYLES = tuple(TRANSFORMS)
PROXY_STYLES = ("posterize", "pixelate", "sketch", "dither", "grid")


def apply_transform(images: np.ndarray, name: str) -> np.ndarray:
    """Apply a named style to an N x H x W x 3 uint8 batch."""
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown style {name!r}; choose from {list(TRANSFORMS)}.")
    if name == "clean":
        return images
    return np.stack(
        [np.asarray(TRANSFORMS[name](Image.fromarray(image, mode="RGB")), dtype=np.uint8) for image in images]
    )
