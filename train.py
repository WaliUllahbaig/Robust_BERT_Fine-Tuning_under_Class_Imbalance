#!/usr/bin/env python
"""
train.py — Fine-tune BERT for Multi-Class Text Classification.

End-to-end training pipeline:
  1. Load & preprocess the Emotion dataset
  2. Compute inverse-frequency class weights
  3. Initialize bert-base-uncased with a classification head
  4. Train with WeightedLossTrainer (custom weighted CE loss)
  5. Evaluate on the test split and persist the best checkpoint

Usage:
    python train.py                         # uses config.yaml defaults
    python train.py --config my_config.yaml # custom config
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import os

import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
)

from data.dataset import load_and_preprocess
from models.trainer import WeightedLossTrainer
from utils.metrics import compute_metrics
from utils.seed import set_seed

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# GPU Profiling Helper
# ──────────────────────────────────────────────────────────────────────────────

def log_gpu_info() -> None:
    """Log available GPU device information and memory."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info("GPU device  : %s", device)
        logger.info("GPU memory  : %.2f GB", mem_total)
    else:
        logger.warning("No CUDA GPU detected — training will run on CPU.")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str = "config.yaml") -> None:
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    imb_cfg = cfg["imbalance"]
    es_cfg = cfg["early_stopping"]
    repro_cfg = cfg["reproducibility"]

    # ── Reproducibility ───────────────────────────────────────────────────
    set_seed(repro_cfg["seed"], deterministic=repro_cfg["deterministic_cudnn"])
    logger.info("Random seed set to %d (deterministic=%s)", repro_cfg["seed"], repro_cfg["deterministic_cudnn"])

    # ── GPU ───────────────────────────────────────────────────────────────
    log_gpu_info()

    # ── Data ──────────────────────────────────────────────────────────────
    tokenized_datasets, tokenizer, class_weights = load_and_preprocess(
        dataset_name=data_cfg["dataset_name"],
        model_name=model_cfg["name"],
        text_col=data_cfg["text_column"],
        label_col=data_cfg["label_column"],
        max_length=data_cfg["max_length"],
        remove_empty=data_cfg["remove_empty"],
        num_labels=data_cfg["num_labels"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Model ─────────────────────────────────────────────────────────────
    label_names = data_cfg["label_names"]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["name"],
        num_labels=data_cfg["num_labels"],
        id2label=id2label,
        label2id=label2id,
    )
    logger.info("Model loaded: %s (%d parameters)", model_cfg["name"], model.num_parameters())

    # ── Training Arguments ────────────────────────────────────────────────
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set TensorBoard logging directory via environment variable (Transformers v5.x)
    os.environ["TENSORBOARD_LOGGING_DIR"] = train_cfg["logging_dir"]

    # Compute warmup_steps from warmup_ratio
    total_train_samples = len(tokenized_datasets["train"])
    steps_per_epoch = total_train_samples // train_cfg["per_device_train_batch_size"]
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        seed=train_cfg["seed"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"] and torch.cuda.is_available(),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        report_to=train_cfg["report_to"],
        dataloader_num_workers=train_cfg["dataloader_num_workers"],
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = []
    if es_cfg["enabled"]:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_cfg["patience"]))
        logger.info("Early stopping enabled (patience=%d).", es_cfg["patience"])

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        class_weights=class_weights if imb_cfg["use_weighted_loss"] else None,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    t0 = time.perf_counter()
    train_result = trainer.train()
    elapsed = time.perf_counter() - t0

    logger.info("Training completed in %.1f s (%.1f min).", elapsed, elapsed / 60)
    logger.info("Train loss : %.4f", train_result.training_loss)

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        logger.info("Peak GPU memory: %.2f GB", peak_mem)

    # ── Evaluate on test set ──────────────────────────────────────────────
    logger.info("Evaluating on test split …")
    test_metrics = trainer.evaluate(tokenized_datasets["test"], metric_key_prefix="test")
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # ── Save ──────────────────────────────────────────────────────────────
    trainer.save_model(str(output_dir / "best"))
    tokenizer.save_pretrained(str(output_dir / "best"))
    logger.info("Best model saved to %s", output_dir / "best")

    # Persist metrics
    metrics_path = output_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved to %s", metrics_path)

    # Persist training time profile
    profile = {
        "training_time_seconds": round(elapsed, 2),
        "training_time_minutes": round(elapsed / 60, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else None,
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "epochs_completed": train_result.global_step,
    }
    with open(output_dir / "training_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("✓ Pipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT for text classification")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
