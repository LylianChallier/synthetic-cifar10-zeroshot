"""Frozen OpenCLIP backbones and an on-disk cache for their image and text embeddings.

Each backbone is encoded once per (split, style): later runs read ``<cache>/<alias>/<key>.npy``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BackboneSpec:
    alias: str
    arch: str
    pretrained: str
    # Input size forced instead of the checkpoint's (open_clip force_image_size). Our images are
    # 32 px, so a large native resolution only adds compute.
    image_size: int | None = None


BACKBONES: dict[str, BackboneSpec] = {
    spec.alias: spec
    for spec in (
        BackboneSpec("pe-l14", "PE-Core-L-14-336", "meta"),
        # PE-Core G was trained with progressive resolution 98-154-224-336-448 (arXiv 2504.13181,
        # Table 18): 224 px is a resolution it has seen, at about 1/4 of the 448 px cost.
        BackboneSpec("pe-g14-224", "PE-Core-bigG-14-448", "meta", image_size=224),
    )
}

_INTERPOLATIONS = {"bicubic", "bilinear", "nearest"}


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def free_device_memory(device: torch.device) -> None:
    """Return cached accelerator memory before loading the next backbone."""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class Backbone:
    """OpenCLIP model with batched on-device preprocessing of tiny uint8 images."""

    def __init__(self, spec: BackboneSpec, device: torch.device):
        import open_clip

        self.spec, self.device = spec, device
        options = (
            {} if spec.image_size is None else {"force_image_size": spec.image_size}
        )
        model, _, _ = open_clip.create_model_and_transforms(
            spec.arch, pretrained=spec.pretrained, **options
        )
        self.model = model.to(device).eval()
        self.tokenizer = open_clip.get_tokenizer(spec.arch)
        cfg = model.visual.preprocess_cfg
        size = cfg["size"]
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        interpolation = cfg.get("interpolation", "bicubic")
        self.interpolation = (
            interpolation if interpolation in _INTERPOLATIONS else "bicubic"
        )
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
        self.logit_scale = float(model.logit_scale.exp())
        self.num_params = sum(
            parameter.numel() for parameter in model.visual.parameters()
        )

    def _autocast(self):
        return torch.autocast(
            self.device.type,
            dtype=torch.float16,
            enabled=self.device.type in ("cuda", "mps"),
        )

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        options = {} if self.interpolation == "nearest" else {"align_corners": False}
        return F.interpolate(
            x, size=self.size, mode=self.interpolation, **options
        ).clamp_(0, 1)

    @torch.inference_mode()
    def encode_images(
        self, images: np.ndarray, batch_size: int, log_every: float = 0.1
    ) -> np.ndarray:
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(
                f"Expected N x H x W x 3 uint8 images, got {images.shape} {images.dtype}."
            )
        chunks, next_log, start_time = [], log_every, time.perf_counter()
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(images[start : start + batch_size]).to(self.device)
            x = self._resize(batch.permute(0, 3, 1, 2).float().div_(255))
            x = (x - self.mean) / self.std
            with self._autocast():
                features = self.model.encode_image(x)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
            done = min(start + batch_size, len(images)) / len(images)
            if done >= next_log:
                elapsed = time.perf_counter() - start_time
                print(
                    f"    {done:4.0%} ({elapsed / done * (1 - done):.0f}s left)",
                    flush=True,
                )
                next_log += log_every
        features = torch.cat(chunks)
        if not torch.isfinite(features).all():
            raise FloatingPointError(f"{self.spec.alias}: non-finite image embeddings.")
        return features.numpy().astype(np.float16)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = self.tokenizer(texts[start : start + batch_size]).to(self.device)
            with self._autocast():
                features = self.model.encode_text(tokens)
            chunks.append(F.normalize(features.float(), dim=-1).cpu())
        return torch.cat(chunks).numpy().astype(np.float32)


class FeatureStore:
    """Cache embeddings per backbone; the model is loaded only on a cache miss."""

    def __init__(
        self,
        spec: BackboneSpec,
        root: Path,
        device: torch.device,
        batch_size: int = 128,
    ):
        self.spec, self.device, self.batch_size = spec, device, batch_size
        self.dir = Path(root) / spec.alias
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir / "meta.json"

    @cached_property
    def backbone(self) -> Backbone:
        start = time.perf_counter()
        backbone = Backbone(self.spec, self.device)
        print(
            f"  loaded {self.spec.arch}/{self.spec.pretrained} in {time.perf_counter() - start:.0f}s",
            flush=True,
        )
        self._update_meta(
            arch=self.spec.arch,
            pretrained=self.spec.pretrained,
            logit_scale=backbone.logit_scale,
            visual_params=backbone.num_params,
            input_size=list(backbone.size),
        )
        return backbone

    @property
    def meta(self) -> dict:
        if not self._meta_path.exists():
            _ = self.backbone  # Loading the model writes the metadata.
        return json.loads(self._meta_path.read_text())

    def _update_meta(self, **values) -> None:
        meta = (
            json.loads(self._meta_path.read_text()) if self._meta_path.exists() else {}
        )
        meta.update(values)
        self._meta_path.write_text(json.dumps(meta, indent=2))

    def images(self, key: str, make_images: Callable[[], np.ndarray]) -> np.ndarray:
        path = self.dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        images = make_images()
        print(f"  encoding {key} ({len(images)} images)", flush=True)
        start = time.perf_counter()
        features = self.backbone.encode_images(images, self.batch_size)
        seconds = time.perf_counter() - start
        self._update_meta(images_per_second=round(len(images) / seconds, 1))
        temporary = path.with_name(f"{path.stem}.tmp.npy")
        np.save(temporary, features)
        temporary.replace(path)
        return features

    def texts(self, prompts: list[str]) -> np.ndarray:
        index_path, array_path = self.dir / "texts.json", self.dir / "texts.npy"
        known: list[str] = (
            json.loads(index_path.read_text()) if index_path.exists() else []
        )
        array = np.load(array_path) if known else None
        known_set = set(known)
        missing = [
            prompt for prompt in dict.fromkeys(prompts) if prompt not in known_set
        ]
        if missing:
            new = self.backbone.encode_texts(missing)
            array = new if array is None else np.concatenate([array, new])
            known += missing
            np.save(array_path, array)
            index_path.write_text(json.dumps(known))
        position = {prompt: row for row, prompt in enumerate(known)}
        return array[[position[prompt] for prompt in prompts]]
