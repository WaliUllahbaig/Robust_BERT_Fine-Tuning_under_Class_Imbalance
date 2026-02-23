"""
Reproducibility utilities.

Sets all random seeds and configures deterministic behaviour for PyTorch/CUDA.

Remaining sources of non-determinism (documented for transparency):
  - CUDA kernel-level floating-point atomics ordering
  - DataLoader worker-level randomness when num_workers > 0
  - cuDNN auto-tuner (disabled via benchmark=False)
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds and optionally enforce deterministic CUDA ops.

    Args:
        seed: Integer seed shared across all RNGs.
        deterministic: If True, force deterministic cuDNN algorithms at the
            cost of a small runtime overhead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch ≥ 1.8 — raise errors on non-deterministic ops
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
