from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BinaryDetectionCounts:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0


def precision(tp: int, fp: int) -> float:
    denominator = tp + fp
    return tp / denominator if denominator else 0.0


def recall(tp: int, fn: int) -> float:
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def f1_score(precision_value: float, recall_value: float) -> float:
    denominator = precision_value + recall_value
    return 2 * precision_value * recall_value / denominator if denominator else 0.0


def summarize_counts(counts: BinaryDetectionCounts) -> dict[str, float | int]:
    p = precision(counts.true_positives, counts.false_positives)
    r = recall(counts.true_positives, counts.false_negatives)
    return {
        "true_positives": counts.true_positives,
        "false_positives": counts.false_positives,
        "false_negatives": counts.false_negatives,
        "precision": p,
        "recall": r,
        "f1_score": f1_score(p, r),
    }

