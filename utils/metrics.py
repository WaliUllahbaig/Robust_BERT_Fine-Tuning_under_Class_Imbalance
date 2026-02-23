"""
Evaluation metrics for multi-class classification.

Provides a HuggingFace Trainer-compatible `compute_metrics` function that
returns accuracy, macro-F1, and per-class precision/recall/F1.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def compute_metrics(eval_pred) -> Dict[str, float]:
    """Compute accuracy, macro-F1, and per-class precision/recall/F1.

    Compatible with ``transformers.Trainer(compute_metrics=...)``.

    Args:
        eval_pred: ``EvalPrediction`` namedtuple with `predictions` and
            `label_ids` arrays.

    Returns:
        Dictionary of metric names → float values.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    metrics: Dict[str, float] = {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "macro_precision": precision_score(labels, preds, average="macro"),
        "macro_recall": recall_score(labels, preds, average="macro"),
    }

    # Per-class breakdown
    precision_per = precision_score(labels, preds, average=None)
    recall_per = recall_score(labels, preds, average=None)
    f1_per = f1_score(labels, preds, average=None)

    for idx, name in enumerate(LABEL_NAMES):
        metrics[f"precision_{name}"] = float(precision_per[idx])
        metrics[f"recall_{name}"] = float(recall_per[idx])
        metrics[f"f1_{name}"] = float(f1_per[idx])

    return metrics
