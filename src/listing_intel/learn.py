"""Learn ranking weights from graded relevance with a tiny numpy logistic regression."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from .match import FEATURES


def fit_weights(
    rows: Sequence[tuple[dict[str, float], int]],
    epochs: int = 400,
    lr: float = 0.8,
    l2: float = 1e-3,
) -> dict[str, float]:
    """Fit P(relevant | features) where relevant means grade >= 2.

    Weights are clipped to be non-negative (every feature is oriented 'more is better') and
    normalised to sum to one, so the result is directly usable as ranking weights and stays
    explainable. Raises ``ValueError`` if there is nothing to learn from.
    """
    if not rows:
        raise ValueError("no training rows")
    x = np.array([[f.get(k, 0.0) for k in FEATURES] for f, _ in rows])
    y = np.array([1.0 if g >= 2 else 0.0 for _, g in rows])
    if y.min() == y.max():
        raise ValueError("training labels are all identical")
    w: npt.NDArray[np.float64] = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        grad = x.T @ (p - y) / len(y) + l2 * w
        w = w - lr * grad
        b -= lr * float(np.mean(p - y))
    clipped = np.clip(w, 0, None)
    total = float(clipped.sum())
    if total == 0:
        raise ValueError("all learned weights are non-positive")
    return {k: float(v / total) for k, v in zip(FEATURES, clipped, strict=True)}
