"""
Custom HuggingFace Trainer with Weighted Cross-Entropy Loss.

Overrides ``compute_loss`` to inject inverse-frequency class weights into
``nn.CrossEntropyLoss``, improving minority-class recall under imbalance.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import Trainer


class WeightedLossTrainer(Trainer):
    """Trainer subclass that applies class-weighted CrossEntropyLoss.

    Usage::

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            class_weights=class_weights,   # <-- extra kwarg
            ...
        )

    The ``class_weights`` tensor is moved to the model's device automatically.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, any]],
        return_outputs: bool = False,
        **kwargs,
    ):
        """Forward pass + weighted cross-entropy loss computation.

        Args:
            model: The wrapped ``PreTrainedModel``.
            inputs: Batch dict with ``input_ids``, ``attention_mask``, ``labels``.
            return_outputs: Whether to return model outputs alongside the loss.

        Returns:
            Loss scalar or ``(loss, outputs)`` tuple.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss
