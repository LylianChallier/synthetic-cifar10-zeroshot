"""Competition data: loading the .npz archives, balanced subsets and submission files."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)  # fmt: skip
REQUIRED_FILES = ("validation.npz", "test.npz", "sample_submission.csv")


def _walk(root: Path, max_depth: int | None = None):
    """``os.walk`` that follows symlinked folders (Kaggle mounts its inputs through symlinks)."""
    for directory, subdirs, files in os.walk(root, followlinks=True):
        if max_depth is not None and len(Path(directory).relative_to(root).parts) >= max_depth:
            subdirs.clear()
        yield Path(directory), subdirs, files


def resolve_data_dir(preferred: str | Path, search_roots=("/kaggle/input",)) -> Path:
    """Return ``preferred`` if it holds the competition files, else the unique match under ``search_roots``."""
    preferred = Path(preferred)
    if all((preferred / name).is_file() for name in REQUIRED_FILES):
        return preferred
    matches = sorted(
        {
            directory
            for root in map(Path, search_roots)
            if root.is_dir()
            for directory, _, files in _walk(root)
            if all(name in files for name in REQUIRED_FILES)
        }
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Competition files {REQUIRED_FILES} not found in {preferred} "
            f"(candidates under {list(search_roots)}: {[str(m) for m in matches]})."
        )
    return matches[0]


class NpzImages:
    """Images (N x 32 x 32 x 3 uint8), ids and optional labels of one .npz archive."""

    def __init__(self, path: str | Path, require_labels: bool = False):
        with np.load(path) as archive:
            if "images" not in archive:
                raise ValueError(f"{path} has no 'images' array.")
            self.images = archive["images"].astype(np.uint8)
            self.ids = archive["ids"].astype(np.int64) if "ids" in archive else np.arange(len(self.images))
            self.labels = archive["labels"].astype(np.int64) if "labels" in archive else None
        if self.images.ndim != 4 or self.images.shape[-1] != 3:
            raise ValueError(f"Expected N x H x W x 3 uint8 images, got {self.images.shape}.")
        if require_labels and self.labels is None:
            raise ValueError(f"{path} has no labels.")
        if self.labels is not None and len(self.labels) != len(self.images):
            raise ValueError("The image and label counts differ.")


def balanced_subset(labels: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    """Sorted indices of ``per_class`` random examples of every class, reproducible with ``seed``."""
    rng = np.random.default_rng(seed)
    picks = []
    for cls in range(len(CLASS_NAMES)):
        members = np.flatnonzero(labels == cls)
        if len(members) < per_class:
            raise ValueError(f"Class {CLASS_NAMES[cls]} has only {len(members)} examples, {per_class} requested.")
        picks.append(rng.choice(members, per_class, replace=False))
    return np.sort(np.concatenate(picks))


def save_submission(ids, labels, sample_csv: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Validate predictions against sample_submission.csv and write them to disk."""
    submission = pd.DataFrame(
        {"id": np.asarray(ids, dtype=np.int64), "label": np.asarray(labels, dtype=np.int64)}
    )
    sample = pd.read_csv(sample_csv)
    if not np.array_equal(sample["id"].to_numpy(), submission["id"].to_numpy()):
        raise ValueError("Submission IDs are not aligned with sample_submission.csv.")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
