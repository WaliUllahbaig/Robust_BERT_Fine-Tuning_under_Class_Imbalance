#!/usr/bin/env python
"""
evaluate.py — Post-Training Evaluation & Error Analysis.

Performs:
  1. Full test-set evaluation (accuracy, macro-F1, per-class P/R/F1)
  2. Confusion matrix visualization (saved as PNG)
  3. Per-class precision / recall bar chart
  4. Systematic error analysis of misclassified examples
  5. Confusion-pair ranking (most frequent error directions)

Usage:
    python evaluate.py                                  # defaults
    python evaluate.py --model_dir ./models/bert-emotion/best
    python evaluate.py --config config.yaml --n_errors 50
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from data.dataset import load_and_preprocess
from utils.metrics import compute_metrics
from utils.seed import set_seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def predict(trainer: Trainer, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, labels) arrays."""
    output = trainer.predict(dataset)
    preds = np.argmax(output.predictions, axis=-1)
    labels = output.label_ids
    return preds, labels


# ──────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: List[str],
    save_path: Path,
) -> None:
    """Generate and save a normalized confusion matrix heatmap."""
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Raw counts
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Normalized
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Oranges",
        xticklabels=label_names, yticklabels=label_names, ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Per-Class Bar Chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_per_class_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: List[str],
    save_path: Path,
) -> None:
    """Bar chart of per-class precision, recall, and F1."""
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=range(len(label_names)),
    )

    x = np.arange(len(label_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label="Precision", color="#4C72B0")
    ax.bar(x, recall, width, label="Recall", color="#DD8452")
    ax.bar(x + width, f1, width, label="F1", color="#55A868")

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision / Recall / F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Per-class metrics chart saved → %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# Error Analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis(
    texts: List[str],
    labels: np.ndarray,
    preds: np.ndarray,
    label_names: List[str],
    n_samples: int = 25,
) -> Dict:
    """Structured error analysis of misclassified examples.

    Returns:
        Dictionary with:
            - ``confusion_pairs``: Counter of (true→pred) error directions
            - ``examples``: List of misclassified sample dicts
            - ``analysis_summary``: Human-readable summary string
    """
    errors = []
    confusion_pairs: Counter = Counter()

    for i in range(len(labels)):
        if preds[i] != labels[i]:
            true_name = label_names[labels[i]]
            pred_name = label_names[preds[i]]
            confusion_pairs[(true_name, pred_name)] += 1
            errors.append({
                "index": int(i),
                "text": texts[i][:300],  # truncate for readability
                "true_label": true_name,
                "predicted_label": pred_name,
            })

    # Top confusion pairs
    top_pairs = confusion_pairs.most_common(10)

    # Sample errors for manual inspection
    sampled = errors[:n_samples]

    # Build summary
    total_errors = len(errors)
    total_samples = len(labels)
    summary_lines = [
        f"Total misclassified: {total_errors} / {total_samples} ({100 * total_errors / total_samples:.1f}%)",
        "",
        "Top confusion pairs (true → predicted : count):",
    ]
    for (true_lbl, pred_lbl), cnt in top_pairs:
        summary_lines.append(f"  {true_lbl:>10s} → {pred_lbl:<10s} : {cnt}")

    summary_lines.extend([
        "",
        "Example failures:",
        "-" * 80,
    ])
    for ex in sampled[:5]:
        summary_lines.append(
            f"  [{ex['true_label']} → {ex['predicted_label']}] \"{ex['text'][:120]}…\""
        )
    summary_lines.append("-" * 80)

    # Failure mode analysis
    summary_lines.extend([
        "",
        "Hypothesized failure modes:",
        "  1. Context length: long texts truncated at max_length lose sentiment cues at the end.",
        "  2. Implicit sentiment: sarcastic or subtle emotional expressions are hard for BERT.",
        "  3. Label ambiguity: emotions like 'love' vs 'joy', or 'anger' vs 'fear' share lexical overlap.",
        "  4. Class imbalance: minority classes ('surprise', 'love') have fewer training examples.",
    ])

    summary = "\n".join(summary_lines)

    return {
        "confusion_pairs": dict(
            {f"{t} → {p}": c for (t, p), c in confusion_pairs.most_common()}
        ),
        "examples": sampled,
        "analysis_summary": summary,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(
    config_path: str = "config.yaml",
    model_dir: str | None = None,
    n_errors: int | None = None,
) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    repro_cfg = cfg["reproducibility"]

    set_seed(repro_cfg["seed"])

    # Resolve paths
    if model_dir is None:
        model_dir = str(Path(cfg["training"]["output_dir"]) / "best")
    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_errors is None:
        n_errors = eval_cfg["error_analysis_samples"]

    label_names = data_cfg["label_names"]

    # ── Load data ─────────────────────────────────────────────────────────
    tokenized_datasets, tokenizer, _ = load_and_preprocess(
        dataset_name=data_cfg["dataset_name"],
        model_name=model_cfg["name"],
        text_col=data_cfg["text_column"],
        label_col=data_cfg["label_column"],
        max_length=data_cfg["max_length"],
        remove_empty=data_cfg["remove_empty"],
        num_labels=data_cfg["num_labels"],
    )
    test_dataset = tokenized_datasets["test"]

    # ── Load model ────────────────────────────────────────────────────────
    logger.info("Loading model from %s", model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    eval_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    data_collator = DataCollatorWithPadding(tokenizer=eval_tokenizer)

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        tokenizer=eval_tokenizer,
        compute_metrics=compute_metrics,
    )

    # ── Predict ───────────────────────────────────────────────────────────
    preds, labels = predict(trainer, test_dataset)

    # ── Metrics ───────────────────────────────────────────────────────────
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    report = classification_report(labels, preds, target_names=label_names, digits=4)

    logger.info("Test Accuracy : %.4f", acc)
    logger.info("Test Macro-F1 : %.4f", macro_f1)
    logger.info("\n%s", report)

    # Save classification report
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"Macro-F1 : {macro_f1:.4f}\n\n")
        f.write(report)
    logger.info("Classification report saved → %s", output_dir / "classification_report.txt")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    if eval_cfg["confusion_matrix"]:
        plot_confusion_matrix(
            labels, preds, label_names,
            save_path=output_dir / "confusion_matrix.png",
        )

    # ── Per-Class Metrics Chart ───────────────────────────────────────────
    if eval_cfg["per_class_metrics"]:
        plot_per_class_metrics(
            labels, preds, label_names,
            save_path=output_dir / "per_class_metrics.png",
        )

    # ── Error Analysis ────────────────────────────────────────────────────
    # Retrieve raw text from the original dataset
    raw_dataset = load_dataset(data_cfg["dataset_name"])
    raw_texts = raw_dataset["test"][data_cfg["text_column"]]

    analysis = error_analysis(raw_texts, labels, preds, label_names, n_samples=n_errors)

    # Print summary
    logger.info("\n%s", analysis["analysis_summary"])

    # Persist
    with open(output_dir / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    logger.info("Error analysis saved → %s", output_dir / "error_analysis.json")

    with open(output_dir / "error_analysis_summary.txt", "w", encoding="utf-8") as f:
        f.write(analysis["analysis_summary"])
    logger.info("Error summary saved → %s", output_dir / "error_analysis_summary.txt")

    logger.info("✓ Evaluation complete. Artefacts in → %s/", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BERT model")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model_dir", type=str, default=None, help="Path to saved model checkpoint")
    parser.add_argument("--n_errors", type=int, default=None, help="Number of error examples to log")
    args = parser.parse_args()
    main(args.config, args.model_dir, args.n_errors)
