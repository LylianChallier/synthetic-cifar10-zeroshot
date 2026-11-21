"""Final classifier: zero-shot PE-Core encoders, optionally mixed with a linear head; writes a submission.

Usage:
  python predict.py --head synthetic --filter-zeroshot   # E016, final system (Kaggle 96.48)
  python predict.py --head none                          # E017, zero-shot only (96.38)
  python predict.py --head cifar                         # E018, head on CIFAR-10 photos (95.88)

Reads only cached embeddings (encode.py, generate_synthetic.py). For each encoder:
  zero-shot head  cosine similarity to the mean embedding of the 80 CLIP templates of each class
  linear head     multinomial logistic regression on standardized embeddings of the --head images
                  (C picked on 20 % held out); --filter-zeroshot keeps the images whose zero-shot
                  prediction matches their label (He et al., ICLR 2023)
Both heads get one temperature fitted on the grayscale validation images, then
  log p = log_softmax((1 - alpha) * t_zs * z_zs + alpha * t_head * z_head),   alpha = 0.5.
The encoders are averaged in probability with equal weights. Every test image is classified on its
own: no statistic of the test set is used. Prints clean, grayscale and proxy validation accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from recvis.backbones import BACKBONES, FeatureStore
from recvis.data import NpzImages, balanced_subset, resolve_data_dir, save_submission
from recvis.heads import fit_head, fit_temperature, log_softmax, prototypes
from recvis.styles import PROXY_STYLES, VAL_STYLES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--head", choices=("none", "cifar", "synthetic"), default="synthetic")
    parser.add_argument("--synthetic-npz", type=Path, default=Path("output/synthetic/sdxlt-p80/images32.npz"))
    parser.add_argument("--filter-zeroshot", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--models", nargs="+", default=["pe-l14", "pe-g14-224"], choices=sorted(BACKBONES))
    parser.add_argument("--Cs", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=Path("cache/features"))
    parser.add_argument("--output", type=Path, default=Path("output/submission.csv"))
    args = parser.parse_args()
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must lie in [0, 1].")
    if args.head == "synthetic" and not args.synthetic_npz.exists():
        parser.error(f"{args.synthetic_npz} does not exist: run generate_synthetic.py first.")
    try:
        args.data = resolve_data_dir(args.data)
    except FileNotFoundError as error:
        parser.error(str(error))
    return args


def load(store: FeatureStore, key: str) -> np.ndarray:
    path = store.dir / f"{key}.npy"
    if not path.exists():
        raise SystemExit(f"{store.spec.alias}: missing cached embeddings {path}; run encode.py.")
    features = np.load(path).astype(np.float32)
    return features / np.linalg.norm(features, axis=1, keepdims=True)


def training_set(args: argparse.Namespace) -> tuple[str, np.ndarray]:
    """Cache key and labels of the images the linear head is trained on."""
    if args.head == "cifar":
        train = NpzImages(args.data / "train.npz", require_labels=True)
        return "train-clean-n1000-s0", train.labels[balanced_subset(train.labels, 1000, 0)]
    return f"synth-{args.synthetic_npz.parent.name}", NpzImages(args.synthetic_npz, require_labels=True).labels


def main() -> None:
    args = parse_args()
    validation = NpzImages(args.data / "validation.npz", require_labels=True)
    labels = validation.labels[balanced_subset(validation.labels, 100, 0)]
    test_ids = NpzImages(args.data / "test.npz").ids
    keys = {f"val_{s}": f"val-{s}-n100-s0" for s in VAL_STYLES} | {"test": "test-none"}
    alpha = 0.0 if args.head == "none" else args.alpha

    members = []
    for alias in args.models:
        store = FeatureStore(BACKBONES[alias], args.cache, torch.device("cpu"))
        feats = {name: load(store, key) for name, key in keys.items()}
        protos = prototypes(store)
        zs = {n: store.meta["logit_scale"] * f @ protos.T for n, f in feats.items()}
        t_zs = fit_temperature(zs["val_grayscale"], labels)
        if args.head == "none":
            members.append({n: np.exp(log_softmax(t_zs * zs[n])) for n in feats})
            print(f"  {alias}: zero-shot only", flush=True)
            continue
        train_key, y_train = training_set(args)
        x_train = load(store, train_key)
        if len(x_train) != len(y_train):
            raise SystemExit(f"{alias}: {len(x_train)} training embeddings, {len(y_train)} labels.")
        kept = (x_train @ protos.T).argmax(1) == y_train if args.filter_zeroshot else np.ones(len(y_train), bool)
        scaler, model, C, held = fit_head(x_train[kept], y_train[kept], args.Cs, seed=0)
        head = {n: model.decision_function(scaler.transform(f)) for n, f in feats.items()}
        t_head = fit_temperature(head["val_grayscale"], labels)
        members.append(
            {n: np.exp(log_softmax((1 - alpha) * t_zs * zs[n] + alpha * t_head * head[n])) for n in feats}
        )
        print(f"  {alias}: {kept.sum()}/{len(kept)} training images, C = {C:g}, held-out acc {held:.4f}", flush=True)

    probs = {n: np.mean([m[n] for m in members], axis=0) for n in keys}
    metrics = {s: float((probs[f"val_{s}"].argmax(1) == labels).mean()) for s in VAL_STYLES}
    metrics["proxy"] = float(np.mean([metrics[s] for s in PROXY_STYLES]))
    save_submission(test_ids, probs["test"].argmax(1), args.data / "sample_submission.csv", args.output)
    args.output.with_suffix(".json").write_text(json.dumps({"head": args.head, "alpha": alpha, **metrics}, indent=2))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}), f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
