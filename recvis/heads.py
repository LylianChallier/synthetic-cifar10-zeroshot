"""Classifier heads on frozen embeddings: zero-shot prototypes, temperature scaling, linear head."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from recvis.backbones import FeatureStore
from recvis.prompts import class_prompts


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(1, keepdims=True))


def prototypes(store: FeatureStore) -> np.ndarray:
    """One normalized prototype per class: the mean embedding of its 80 prompts (CLIP ensembling)."""
    per_class = class_prompts()
    flat = store.texts([prompt for prompts in per_class for prompt in prompts])
    bounds = np.cumsum([0] + [len(prompts) for prompts in per_class])
    centroids = np.stack([flat[a:b].mean(0) for a, b in pairwise(bounds)])
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, scale: float = 1.0) -> float:
    """Mean negative log-likelihood of softmax(scale * logits)."""
    return float(-log_softmax(scale * logits)[np.arange(len(labels)), labels].mean())


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Multiplier s of the logits minimizing the validation NLL (Guo et al., ICML 2017); s = 1 / T.

    The NLL is convex in s: a log-spaced grid brackets the minimum, golden-section search refines it.
    """
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite to fit a temperature.")
    grid = np.logspace(-3, 3, 121)
    best = int(np.argmin([nll(logits, labels, s) for s in grid]))
    low, high = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
    ratio = (np.sqrt(5) - 1) / 2
    for _ in range(60):
        a, b = high - ratio * (high - low), low + ratio * (high - low)
        if nll(logits, labels, a) < nll(logits, labels, b):
            high = b
        else:
            low = a
    return float((low + high) / 2)


def fit_head(
    x: np.ndarray, y: np.ndarray, Cs: list[float], seed: int
) -> tuple[StandardScaler, LogisticRegression, float, float]:
    """Logistic regression with C picked on a held-out 20 % (ties: smaller C), refitted on all data."""
    held = np.random.default_rng(seed).permutation(len(y)) % 5 == 0
    scores = []
    for C in Cs:
        scaler = StandardScaler().fit(x[~held])
        model = LogisticRegression(C=C, max_iter=3000).fit(scaler.transform(x[~held]), y[~held])
        scores.append(float((model.predict(scaler.transform(x[held])) == y[held]).mean()))
    best = Cs[int(np.argmax(scores))]
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=best, max_iter=3000).fit(scaler.transform(x), y)
    return scaler, model, best, max(scores)
