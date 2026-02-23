"""
Data loading and preprocessing pipeline.

Handles:
  - Loading the Emotion dataset from HuggingFace Hub
  - Custom text preprocessing (cleaning, empty-sample removal)
  - Tokenization with truncation and dynamic padding
  - Inverse-frequency class-weight computation for weighted loss
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------------------------------

def clean_text(example: Dict) -> Dict:
    """Light-weight text normalization.

    BERT's WordPiece tokenizer already handles most casing via uncased models.
    We strip leading/trailing whitespace and collapse internal whitespace.
    """
    text = example["text"]
    text = " ".join(text.strip().split())  # collapse whitespace
    example["text"] = text
    return example


def remove_empty_samples(dataset_dict: DatasetDict, text_col: str = "text") -> DatasetDict:
    """Drop rows where the text column is empty or whitespace-only."""
    def _is_nonempty(example: Dict) -> bool:
        return bool(example[text_col] and example[text_col].strip())

    before = {split: len(ds) for split, ds in dataset_dict.items()}
    dataset_dict = DatasetDict(
        {split: ds.filter(_is_nonempty) for split, ds in dataset_dict.items()}
    )
    after = {split: len(ds) for split, ds in dataset_dict.items()}
    for split in dataset_dict:
        removed = before[split] - after[split]
        if removed > 0:
            logger.info("Removed %d empty samples from '%s' split.", removed, split)
    return dataset_dict


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def tokenize_factory(
    tokenizer: PreTrainedTokenizerBase,
    text_col: str = "text",
    max_length: int = 128,
):
    """Return a map-ready tokenization function (closure over tokenizer)."""

    def _tokenize(examples: Dict) -> Dict:
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=max_length,
            # Padding is done dynamically by DataCollatorWithPadding at batch time
        )

    return _tokenize


# ---------------------------------------------------------------------------
# Class Weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
) -> torch.Tensor:
    r"""Compute inverse-frequency class weights.

    .. math::

        w_c = \frac{N}{K \cdot n_c}

    where *N* is the total number of samples, *K* the number of classes, and
    *n_c* the count of class *c*.

    Args:
        labels: 1-D array of integer labels.
        num_classes: Total number of classes (K).

    Returns:
        Float tensor of shape ``(num_classes,)`` with per-class weights.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 1)  # guard against missing classes
        w_c = total / (num_classes * n_c)
        weights.append(w_c)
    weights = torch.tensor(weights, dtype=torch.float32)
    logger.info("Class weights: %s", weights.tolist())
    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_preprocess(
    dataset_name: str = "dair-ai/emotion",
    model_name: str = "bert-base-uncased",
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = 128,
    remove_empty: bool = True,
    num_labels: int = 6,
) -> Tuple[DatasetDict, PreTrainedTokenizerBase, Optional[torch.Tensor]]:
    """End-to-end data preparation pipeline.

    1. Load dataset from HuggingFace Hub
    2. Clean text
    3. Remove empty samples (optional)
    4. Tokenize
    5. Compute class weights

    Returns:
        (tokenized_datasets, tokenizer, class_weights)
    """
    logger.info("Loading dataset '%s' …", dataset_name)
    raw_datasets = load_dataset(dataset_name)

    # --- Clean ---
    raw_datasets = raw_datasets.map(clean_text, desc="Cleaning text")

    # --- Remove empties ---
    if remove_empty:
        raw_datasets = remove_empty_samples(raw_datasets, text_col)

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok_fn = tokenize_factory(tokenizer, text_col, max_length)
    tokenized = raw_datasets.map(tok_fn, batched=True, desc="Tokenizing")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", label_col])

    # --- Class weights ---
    train_labels = np.array(tokenized["train"][label_col])
    class_weights = compute_class_weights(train_labels, num_labels)

    # --- Log split sizes ---
    for split, ds in tokenized.items():
        logger.info("Split %-12s : %d samples", split, len(ds))

    return tokenized, tokenizer, class_weights
