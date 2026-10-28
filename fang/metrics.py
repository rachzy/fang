"""Threshold selection and the reported metric set.

Metrics are weighted so every star contributes equally, matching the fitting
weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from .data import star_balanced_weights

EPS = 1e-12


def _as_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"Expected {n} weights, got {w.shape}.")
    return w


def _both_classes(y: np.ndarray) -> bool:
    return y.size > 0 and np.unique(y).size == 2


def _broadcast_thresholds(threshold: float | np.ndarray, n: int) -> np.ndarray:
    """Accept one threshold or one per row, always returning ``n`` of them."""
    values = np.asarray(threshold, dtype=float)
    try:
        return np.broadcast_to(values, (n,)).copy()
    except ValueError as exc:
        raise ValueError(
            f"Expected a scalar threshold or {n} thresholds, got shape {values.shape}."
        ) from exc


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdChoice:
    """The highest-recall operating point satisfying a precision floor."""

    threshold: float
    precision: float
    recall: float
    min_precision: float
    achieved: bool
    n_candidates: int
    n_resamples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": float(self.threshold),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "min_precision": float(self.min_precision),
            "precision_floor_met": bool(self.achieved),
            "n_candidate_thresholds": int(self.n_candidates),
            "n_resamples": int(getattr(self, "n_resamples", 0)),
        }


def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
    sample_weight: np.ndarray | None = None,
) -> ThresholdChoice:
    """Choose the highest-recall threshold satisfying ``min_precision``.

    Recall ties resolve toward higher precision and then the higher threshold.
    When no threshold reaches the floor, the highest-precision one is returned
    with ``achieved`` false.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    w = _as_weights(sample_weight, y.size)
    if y.size == 0:
        raise ValueError("Cannot select a threshold from an empty set of candidates.")

    positive = w[y == 1].sum()
    if positive <= 0:
        raise ValueError("Cannot select a threshold: no positive candidates present.")

    # Walk the scores from the top: cumulative weights at the last row of each
    # distinct score are the counts flagged by that score used as a threshold.
    order = np.argsort(-p, kind="stable")
    ordered = p[order]
    last_of_value = np.r_[ordered[1:] != ordered[:-1], True]
    candidates = ordered[last_of_value]
    tp = np.cumsum(w[order] * (y[order] == 1))[last_of_value]
    fp = np.cumsum(w[order] * (y[order] == 0))[last_of_value]
    recall = tp / positive
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)

    feasible = np.flatnonzero((tp > 0) & (precision >= min_precision - EPS))
    achieved = feasible.size > 0
    if achieved:
        order = sorted(
            feasible.tolist(),
            key=lambda i: (-recall[i], -precision[i], -candidates[i]),
        )
    else:
        order = sorted(
            range(candidates.size),
            key=lambda i: (-precision[i], -recall[i], -candidates[i]),
        )
    best = order[0]
    return ThresholdChoice(
        threshold=float(candidates[best]),
        precision=float(precision[best]),
        recall=float(recall[best]),
        min_precision=float(min_precision),
        achieved=bool(achieved),
        n_candidates=int(candidates.size),
    )


def bootstrap_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    star_id: np.ndarray,
    min_precision: float,
    n_resamples: int,
    seed: int = 42,
    weighted: bool = True,
) -> ThresholdChoice:
    """Median of the precision-floor thresholds over star-bootstrap resamples.

    Near the top of the ranking the precision curve rests on a handful of
    rows, so the single-sample threshold jumps with any one of them. Resampling
    whole stars and taking the median cutoff steadies it. Resamples that cannot
    reach the floor are skipped; with none left, or ``n_resamples`` 0, the
    single-sample choice is returned.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    stars = np.asarray(star_id, dtype=object)
    full_weights = star_balanced_weights(stars) if weighted else None
    point = select_threshold(y, p, min_precision, sample_weight=full_weights)
    if n_resamples <= 0:
        return point

    unique_stars = np.array(sorted(set(stars.tolist())), dtype=object)
    index_by_star = {star: np.flatnonzero(stars == star) for star in unique_stars}
    rng = np.random.default_rng(seed)
    thresholds: list[float] = []
    for _ in range(n_resamples):
        picked = rng.integers(0, unique_stars.size, size=unique_stars.size)
        rows = np.concatenate([index_by_star[unique_stars[i]] for i in picked])
        if not (y[rows] == 1).any():
            continue
        replicate_ids = np.concatenate(
            [np.full(index_by_star[unique_stars[s]].size, f"{s}#{k}", dtype=object)
             for k, s in enumerate(picked)]
        )
        weights = star_balanced_weights(replicate_ids) if weighted else None
        choice = select_threshold(y[rows], p[rows], min_precision, sample_weight=weights)
        if choice.achieved:
            thresholds.append(choice.threshold)
    if not thresholds:
        return point

    threshold = float(np.median(thresholds))
    w = _as_weights(full_weights, y.size)
    flagged = p >= threshold
    tp = w[flagged & (y == 1)].sum()
    fp = w[flagged & (y == 0)].sum()
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return ThresholdChoice(
        threshold=threshold,
        precision=precision,
        recall=float(tp / w[y == 1].sum()),
        min_precision=float(min_precision),
        achieved=bool(tp > 0 and precision >= min_precision - EPS),
        n_candidates=point.n_candidates,
        n_resamples=len(thresholds),
    )


def recall_at_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Best recall over thresholds keeping precision >= ``min_precision``.

    Threshold-free, like average precision: it scores the ranking in the
    high-precision region. 0 when no threshold reaches the floor, NaN when the
    subset has only one class.
    """
    y = np.asarray(y_true, dtype=int)
    if not _both_classes(y):
        return float("nan")
    choice = select_threshold(y, y_prob, min_precision, sample_weight=sample_weight)
    return choice.recall if choice.achieved else 0.0


# ---------------------------------------------------------------------------
# Metric set
# ---------------------------------------------------------------------------
def safe_average_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    """Average precision, or NaN when the subset has only one class."""
    y = np.asarray(y_true, dtype=int)
    if not _both_classes(y):
        return float("nan")
    return float(
        average_precision_score(y, np.asarray(y_prob, dtype=float), sample_weight=sample_weight)
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | np.ndarray,
    sample_weight: np.ndarray | None = None,
    min_precision: float | None = None,
) -> dict[str, Any]:
    """Full metric set for one model at one operating point.

    ``threshold`` may be an array when pooling rows that were scored under
    different fold-specific operating points. ``recall_at_precision`` ignores
    it and is NaN unless ``min_precision`` is given.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    w = _as_weights(sample_weight, y.size)
    thresholds = _broadcast_thresholds(threshold, y.size)
    pred = (p >= thresholds).astype(int)

    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    tn = w[(pred == 0) & (y == 0)].sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (4 * precision + recall) == 0:
        f2 = float("nan")
    else:
        f2 = 5.0 * precision * recall / (4.0 * precision + recall)

    both = _both_classes(y)
    clipped = np.clip(p, EPS, 1.0 - EPS)
    return {
        "n": int(y.size),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "threshold": float(thresholds[0]) if np.unique(thresholds).size == 1 else None,
        "threshold_mean": float(thresholds.mean()),
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "precision": float(precision),
        "recall": float(recall),
        "f2": float(f2),
        "average_precision": float(average_precision_score(y, p, sample_weight=w))
        if both
        else float("nan"),
        "recall_at_precision": recall_at_precision(y, p, min_precision, sample_weight=w)
        if min_precision is not None
        else float("nan"),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=w)) if both else float("nan"),
        "brier_score": float(np.average((p - y) ** 2, weights=w)),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1], sample_weight=w))
        if both
        else float("nan"),
        "confusion_matrix": {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
        },
        "confusion_matrix_counts": {
            "tp": int(((pred == 1) & (y == 1)).sum()),
            "fp": int(((pred == 1) & (y == 0)).sum()),
            "fn": int(((pred == 0) & (y == 1)).sum()),
            "tn": int(((pred == 0) & (y == 0)).sum()),
        },
        "n_flagged": int(pred.sum()),
    }


# ---------------------------------------------------------------------------
# Calibration and per-star views
# ---------------------------------------------------------------------------
def calibration_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    sample_weight: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """Weighted reliability curve over equal-width probability bins."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    w = _as_weights(sample_weight, y.size)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    points: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = index == b
        total = w[mask].sum()
        if total <= 0:
            continue
        points.append(
            {
                "bin_lower": float(edges[b]),
                "bin_upper": float(edges[b + 1]),
                "mean_predicted": float(np.average(p[mask], weights=w[mask])),
                "observed_rate": float(np.average(y[mask], weights=w[mask])),
                "weight": float(total),
                "count": int(mask.sum()),
            }
        )
    return points


def per_star_summary(
    star_id: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision: np.ndarray,
) -> list[dict[str, Any]]:
    """One row per star: support, flags, and the recall/precision it contributes."""
    stars = np.asarray(star_id, dtype=object)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    d = np.asarray(decision, dtype=int)

    rows: list[dict[str, Any]] = []
    for star in sorted(set(stars.tolist())):
        mask = stars == star
        tp = int(((d == 1) & (y == 1) & mask).sum())
        fp = int(((d == 1) & (y == 0) & mask).sum())
        fn = int(((d == 0) & (y == 1) & mask).sum())
        tn = int(((d == 0) & (y == 0) & mask).sum())
        rows.append(
            {
                "star_id": star,
                "n_candidates": int(mask.sum()),
                "n_positive": int((y[mask] == 1).sum()),
                "n_flagged": int(d[mask].sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
                "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
                "mean_probability": float(p[mask].mean()),
                "max_probability": float(p[mask].max()),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Star-level bootstrap
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """Percentile confidence intervals from resampling whole stars."""

    intervals: dict[str, dict[str, float]] = field(default_factory=dict)
    n_resamples: int = 0
    confidence_level: float = 0.95

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_resamples": self.n_resamples,
            "confidence_level": self.confidence_level,
            "intervals": self.intervals,
        }


_BOOTSTRAP_METRICS = (
    "precision",
    "recall",
    "f2",
    "average_precision",
    "recall_at_precision",
    "roc_auc",
    "brier_score",
    "log_loss",
)


def star_bootstrap_intervals(
    star_id: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: np.ndarray | float,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
    weighted: bool = True,
    min_precision: float | None = None,
) -> BootstrapResult:
    """Resample stars with replacement and take percentile intervals.

    Stars are the independent unit here: candidates from one star share a light
    curve and a detection pipeline run, so resampling rows would understate the
    uncertainty.
    """
    stars = np.asarray(star_id, dtype=object)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_prob, dtype=float)
    thr = _broadcast_thresholds(threshold, y.size)

    unique_stars = np.array(sorted(set(stars.tolist())), dtype=object)
    index_by_star = {star: np.flatnonzero(stars == star) for star in unique_stars}
    rng = np.random.default_rng(seed)

    draws: dict[str, list[float]] = {
        name: []
        for name in _BOOTSTRAP_METRICS
        if name != "recall_at_precision" or min_precision is not None
    }
    for _ in range(n_resamples):
        picked = rng.integers(0, unique_stars.size, size=unique_stars.size)
        rows = np.concatenate([index_by_star[unique_stars[i]] for i in picked])
        # Weights are rebuilt per resample so duplicated stars stay equal-weight.
        replicate_ids = np.concatenate(
            [np.full(index_by_star[unique_stars[s]].size, f"{s}#{k}", dtype=object)
             for k, s in enumerate(picked)]
        )
        w = star_balanced_weights(replicate_ids) if weighted else None
        ys, ps, ts = y[rows], p[rows], thr[rows]
        if not _both_classes(ys):
            continue
        pred = (ps >= ts).astype(int)
        ww = _as_weights(w, ys.size)
        tp = ww[(pred == 1) & (ys == 1)].sum()
        fp = ww[(pred == 1) & (ys == 0)].sum()
        fn = ww[(pred == 0) & (ys == 1)].sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f2 = (
            5.0 * precision * recall / (4.0 * precision + recall)
            if not (np.isnan(precision) or np.isnan(recall) or (4 * precision + recall) == 0)
            else float("nan")
        )
        clipped = np.clip(ps, EPS, 1 - EPS)
        draws["precision"].append(float(precision))
        draws["recall"].append(float(recall))
        draws["f2"].append(float(f2))
        draws["average_precision"].append(
            float(average_precision_score(ys, ps, sample_weight=ww))
        )
        if min_precision is not None:
            draws["recall_at_precision"].append(
                recall_at_precision(ys, ps, min_precision, sample_weight=ww)
            )
        draws["roc_auc"].append(float(roc_auc_score(ys, ps, sample_weight=ww)))
        draws["brier_score"].append(float(np.average((ps - ys) ** 2, weights=ww)))
        draws["log_loss"].append(float(log_loss(ys, clipped, labels=[0, 1], sample_weight=ww)))

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, dict[str, float]] = {}
    for name, values in draws.items():
        arr = np.asarray(values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            intervals[name] = {"lower": float("nan"), "upper": float("nan"), "n_valid": 0}
            continue
        intervals[name] = {
            "lower": float(np.quantile(arr, alpha)),
            "upper": float(np.quantile(arr, 1.0 - alpha)),
            "median": float(np.median(arr)),
            "n_valid": int(arr.size),
        }
    return BootstrapResult(
        intervals=intervals, n_resamples=n_resamples, confidence_level=confidence_level
    )
