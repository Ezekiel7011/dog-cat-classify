from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    false_call_rate_by_class: dict[str, float]
    miss_rate_by_class: dict[str, float]
    confusion_matrix: list[list[int]]
    per_class: dict[str, dict[str, float]]


def compute_metrics(y_true: list[int], y_pred: list[int], classes: list[str]) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("At least one sample is required to compute metrics")

    matrix = [[0 for _ in classes] for _ in classes]
    for actual, predicted in zip(y_true, y_pred):
        if actual < 0 or actual >= len(classes) or predicted < 0 or predicted >= len(classes):
            raise ValueError("Label index is outside the configured class range")
        matrix[actual][predicted] += 1

    total = sum(sum(row) for row in matrix)
    correct = sum(matrix[index][index] for index in range(len(classes)))
    false_call_rate: dict[str, float] = {}
    miss_rate: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}

    for idx, class_name in enumerate(classes):
        tp = matrix[idx][idx]
        predicted_positive = sum(row[idx] for row in matrix)
        actual_positive = sum(matrix[idx])
        fp = predicted_positive - tp
        fn = actual_positive - tp
        actual_negative = total - actual_positive
        precision = tp / predicted_positive if predicted_positive else 0.0
        recall = tp / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        false_call_rate[class_name] = fp / actual_negative if actual_negative else 0.0
        miss_rate[class_name] = fn / actual_positive if actual_positive else 0.0
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(actual_positive),
        }

    macro_precision = sum(item["precision"] for item in per_class.values()) / len(classes)
    macro_recall = sum(item["recall"] for item in per_class.values()) / len(classes)
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(classes)

    return ClassificationMetrics(
        accuracy=correct / total,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        false_call_rate_by_class=false_call_rate,
        miss_rate_by_class=miss_rate,
        confusion_matrix=matrix,
        per_class=per_class,
    )
