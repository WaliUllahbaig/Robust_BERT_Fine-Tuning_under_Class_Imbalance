# Robust BERT Fine-Tuning under Class Imbalance Imbalance Fine-

> **Fine-tuning `bert-base-uncased` for multi-class emotion classification with inverse-frequency weighted cross-entropy loss, systematic error analysis, and full reproducibility controls.**

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Dataset](#dataset)
5. [Modeling Decisions](#modeling-decisions)
6. [Hyperparameters](#hyperparameters)
7. [Loss Function & Math](#loss-function--math)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation Metrics](#evaluation-metrics)
10. [Error Analysis](#error-analysis)
11. [Failure Modes](#failure-modes)
12. [Class Imbalance Handling](#class-imbalance-handling)
13. [Regularization](#regularization)
14. [Supervision Type](#supervision-type)
15. [Compute & Profiling](#compute--profiling)
16. [Reproducibility](#reproducibility)
17. [Bias & Ethical Considerations](#bias--ethical-considerations)
18. [Results](#results)
19. [References](#references)

---

## Project Overview

This project fine-tunes a pre-trained **BERT** (`bert-base-uncased`) model for **multi-class text classification** on the [Emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset. The primary research focus is on **handling class imbalance** through inverse-frequency weighted cross-entropy loss, and conducting rigorous **error analysis** to understand model failure modes.

### Key Contributions

- **Custom weighted loss**: Override of the HuggingFace `Trainer.compute_loss()` with class-frequency-aware `nn.CrossEntropyLoss`.
- **Systematic error analysis**: Automated extraction of misclassified examples, confusion pair ranking, and failure mode categorization.
- **Full reproducibility**: Deterministic seed control across Python, NumPy, PyTorch, and CUDA backends.
- **Research-grade evaluation**: Macro-F1 as the primary metric, per-class precision/recall breakdown, and normalized confusion matrices.

---

## Project Structure

```
Robust BERT Fine-Tuning under Class Imbalance/
├── config.yaml              # All hyperparameters and pipeline settings
├── requirements.txt         # Python dependencies
├── train.py                 # End-to-end training pipeline
├── evaluate.py              # Post-training evaluation & error analysis
├── README.md                # This file
│
├── data/
│   ├── __init__.py
│   └── dataset.py           # Data loading, preprocessing, class weights
│
├── models/
│   ├── __init__.py
│   └── trainer.py           # WeightedLossTrainer (custom Trainer subclass)
│
├── utils/
│   ├── __init__.py
│   ├── seed.py              # Reproducibility utilities
│   └── metrics.py           # Evaluation metric computation
│
├── logs/                    # TensorBoard logs
└── experiments/             # Evaluation artefacts (plots, reports, error analysis)
```

---

## Setup & Installation

### Prerequisites

- [Anaconda / Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- NVIDIA GPU with CUDA 12.1+ (recommended) or CPU fallback

### Environment Setup

```bash
# 1. Create conda environment
conda create -n bert python=3.10 -y
conda activate bert

# 2. Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install remaining dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Train
python train.py --config config.yaml

# Evaluate
python evaluate.py --config config.yaml

# TensorBoard
tensorboard --logdir ./logs
```

---

## Dataset

**Emotion** ([dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)) — a multi-class text classification dataset derived from English Twitter data.

| Property | Value |
|---|---|
| **Classes** | 6 (sadness, joy, love, anger, fear, surprise) |
| **Train** | 16,000 |
| **Validation** | 2,000 |
| **Test** | 2,000 |
| **Label type** | Single-label, multi-class |
| **Imbalance** | Moderate (joy ≈ 5,362 vs surprise ≈ 572) |

### Class Distribution

| Label | Train Count | Proportion | Weight ($w_c$) |
|---|---|---|---|
| sadness | ~4,666 | 29.2% | ~0.57 |
| joy | ~5,362 | 33.5% | ~0.50 |
| love | ~1,304 | 8.2% | ~2.04 |
| anger | ~2,159 | 13.5% | ~1.23 |
| fear | ~1,937 | 12.1% | ~1.38 |
| surprise | ~572 | 3.6% | ~4.66 |

> The ~9:1 ratio between the most and least frequent classes (`joy` vs `surprise`) motivates the use of weighted loss.

### Preprocessing Pipeline

1. **Whitespace normalization** — collapse multiple spaces, strip edges
2. **Empty sample removal** — filter rows with empty or whitespace-only text
3. **Tokenization** — `AutoTokenizer` from `bert-base-uncased` with `truncation=True`, `max_length=128`
4. **Dynamic padding** — `DataCollatorWithPadding` pads each batch to the longest sequence (no fixed-length waste)
5. **Label encoding** — integer labels mapped to human-readable names via `id2label` / `label2id`

---

## Modeling Decisions

### Why BERT over BiLSTM?

| Criterion | BERT | BiLSTM |
|---|---|---|
| **Pre-training** | Bidirectional masked LM on BooksCorpus + Wikipedia (3.3B tokens) | None — trained from scratch |
| **Contextual representations** | Deep bidirectional attention captures long-range dependencies | Limited by hidden state bottleneck |
| **Transfer learning** | Strong zero-/few-shot generalization to downstream tasks | Requires large task-specific corpora |
| **Sub-word tokenization** | WordPiece handles OOV words gracefully | Word-level embeddings struggle with OOV |
| **Fine-tuning cost** | ~15 min on a single GPU for the Emotion dataset | Similar wall-clock but weaker performance |

**Decision**: BERT's pre-trained representations dramatically reduce the data requirement for minority classes, which is critical under class imbalance.

### Why Weighted Loss over Oversampling?

| Approach | Pros | Cons |
|---|---|---|
| **Weighted CE Loss** | No data duplication; gradient-level correction; cleanly integrates with Trainer | Sensitive to weight magnitude |
| **Random Oversampling** | Simple implementation | Increases training time; risk of overfitting on duplicated minority samples |
| **SMOTE** | Generates synthetic samples | Not well-defined for high-dimensional transformer embeddings |
| **Focal Loss** | Down-weights easy examples | Additional hyperparameter ($\gamma$); less interpretable |

**Decision**: Weighted cross-entropy provides a clean, principled correction without data augmentation artifacts. Weight magnitudes are derived directly from inverse class frequencies.

### Why Macro-F1 over Accuracy?

Under class imbalance, **accuracy is misleading** — a model predicting only `joy` (the majority class) achieves ~33.5% accuracy while completely ignoring 5 other classes.

**Macro-F1** computes F1 per class and averages uniformly, giving **equal importance to minority classes**:

$$\text{Macro-F1} = \frac{1}{K} \sum_{c=1}^{K} F_1^{(c)}$$

This is the standard metric for imbalanced classification in NLP research.

---

## Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| **Learning rate** | `2e-5` | Standard for BERT fine-tuning (Devlin et al., 2019). Most impactful hyperparameter. |
| **Batch size** | 32 (train) / 64 (eval) | Balances gradient noise and GPU memory. |
| **Epochs** | 5 | Sufficient for convergence on 16K samples; early stopping prevents over-training. |
| **Weight decay** | 0.01 | L2 regularization to prevent overfitting the classification head. |
| **Warmup ratio** | 0.06 | Gradual LR ramp-up stabilizes early training; ~6% of total steps. |
| **LR scheduler** | Linear decay | Monotonically decreasing after warmup; standard for transformer fine-tuning. |
| **Max sequence length** | 128 tokens | Covers >95% of Emotion dataset sentences. |
| **FP16** | Enabled | Mixed-precision: ~2x speedup and ~40% memory reduction on Ampere+ GPUs. |
| **Gradient accumulation** | 1 | Effective batch size = 32; no accumulation needed at this scale. |

### Sensitivity Analysis

**Learning rate** is the single most impactful hyperparameter for BERT fine-tuning:

| Learning Rate | Expected Macro-F1 | Notes |
|---|---|---|
| `1e-5` | ~0.88 | Slow convergence; may underfit |
| `2e-5` | ~0.91 | **Optimal balance** |
| `3e-5` | ~0.90 | Slightly unstable training |
| `5e-5` | ~0.87 | Catastrophic forgetting risk |

---

## Loss Function & Math

### Standard Cross-Entropy

For a single sample with true class $c$ and predicted probability $\hat{y}_c$:

$$\mathcal{L}_{\text{CE}} = -\log(\hat{y}_c) = -\log\left(\frac{e^{z_c}}{\sum_{j=1}^{K} e^{z_j}}\right)$$

where $z_j$ are the raw logits from the classification head.

### Weighted Cross-Entropy

To compensate for class imbalance, we assign a weight $w_c$ to each class:

$$\mathcal{L}_{\text{WCE}} = -\sum_{c=1}^{C} w_c \, y_c \, \log(\hat{y}_c)$$

where $y_c \in \{0, 1\}$ is the one-hot label indicator.

### Inverse-Frequency Weighting

Class weights are computed from training set statistics:

$$w_c = \frac{N}{K \cdot n_c}$$

| Symbol | Meaning |
|---|---|
| $N$ | Total number of training samples |
| $K$ | Number of classes |
| $n_c$ | Number of training samples in class $c$ |

**Properties**:
- Majority classes ($n_c$ large) → $w_c < 1$ → reduced gradient contribution
- Minority classes ($n_c$ small) → $w_c > 1$ → amplified gradient contribution
- If all classes are balanced ($n_c = N/K$), then $w_c = 1$ for all $c$

### Gradient Effect

For logit $z_c$, the gradient of weighted CE with respect to the model parameters $\theta$:

$$\frac{\partial \mathcal{L}_{\text{WCE}}}{\partial \theta} = w_c \left(\hat{y}_c - y_c\right) \frac{\partial z_c}{\partial \theta}$$

The weight $w_c$ **scales the gradient magnitude** for class $c$, ensuring the model receives stronger learning signals from underrepresented classes.

### Implementation

```python
# In models/trainer.py
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        weight = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        loss = loss_fn(logits, labels)
        
        return (loss, outputs) if return_outputs else loss
```

---

## Training Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Load Data  │────▶│  Preprocess  │────▶│   Tokenize    │
│ (HF Hub)    │     │  & Clean     │     │ (WordPiece)   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                  │
                    ┌──────────────┐     ┌────────▼────────┐
                    │  Compute     │     │  Load BERT +    │
                    │  Class Wts   │────▶│  Classification │
                    └──────────────┘     │  Head           │
                                         └────────┬────────┘
                                                   │
                    ┌──────────────┐     ┌─────────▼───────┐
                    │  Save Best   │◀────│ WeightedLoss    │
                    │  Checkpoint  │     │ Trainer.train() │
                    └──────┬───────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  Evaluate    │
                    │  Test Split  │
                    └──────────────┘
```

### Running Training

```bash
# Default configuration
python train.py

# Custom config
python train.py --config experiments/my_config.yaml
```

### Outputs

| Artefact | Path |
|---|---|
| Best model checkpoint | `models/bert-emotion/best/` |
| Test metrics (JSON) | `models/bert-emotion/test_metrics.json` |
| Training profile | `models/bert-emotion/training_profile.json` |
| TensorBoard logs | `logs/` |

---

## Evaluation Metrics

### Primary Metric: Macro-F1

$$\text{Macro-F1} = \frac{1}{K} \sum_{c=1}^{K} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$

where precision $P_c$ and recall $R_c$ for class $c$:

$$P_c = \frac{TP_c}{TP_c + FP_c}, \qquad R_c = \frac{TP_c}{TP_c + FN_c}$$

### Full Metric Suite

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness (majority-biased) |
| **Macro-F1** | Balanced performance across all classes (primary) |
| **Per-class Precision** | How many predictions for class $c$ are correct |
| **Per-class Recall** | How many actual class $c$ samples are found |
| **Confusion Matrix** | Visualize systematic error patterns |

### Running Evaluation

```bash
python evaluate.py --config config.yaml
python evaluate.py --model_dir ./models/bert-emotion/best --n_errors 50
```

### Evaluation Outputs

| Artefact | Path |
|---|---|
| Classification report | `experiments/classification_report.txt` |
| Confusion matrix (PNG) | `experiments/confusion_matrix.png` |
| Per-class chart (PNG) | `experiments/per_class_metrics.png` |
| Error analysis (JSON) | `experiments/error_analysis.json` |
| Error summary (TXT) | `experiments/error_analysis_summary.txt` |

---

## Error Analysis

### Methodology

1. **Extract** all misclassified test examples
2. **Rank confusion pairs** — which (true → predicted) directions are most frequent
3. **Sample misclassified examples** for qualitative inspection
4. **Categorize failure modes** into systematic patterns

### Example Failures

| True Label | Predicted | Text (truncated) | Likely Cause |
|---|---|---|---|
| surprise | joy | "i can't believe how amazing this turned out" | Lexical overlap ("amazing") biases toward joy |
| anger | fear | "i'm trembling with rage at what happened" | "trembling" is a fear-associated token |
| love | joy | "feeling grateful for everything you do" | Gratitude shares semantic space with joy |
| sadness | anger | "sick of being ignored by everyone" | Frustration lexicon overlaps with anger |

### Confusion Pair Analysis

Top confusion directions (expected):

```
     love → joy        : most frequent (semantic overlap)
  surprise → joy       : "amazed/excited" ambiguity
    anger → sadness    : frustrated vs. sad
     fear → sadness    : anxious vs. sad
  surprise → fear      : shock vs. anxiety
```

---

## Failure Modes

### 1. Token Truncation

With `max_length=128`, sentences exceeding ~100 words lose their tail tokens. Sentiment cues appearing at the end of long texts are dropped.

**Mitigation**: Increase `max_length` (at memory cost) or use a sliding window approach.

### 2. Implicit & Sarcastic Sentiment

BERT struggles with sarcasm, irony, and implicit emotion:

> "Oh great, another Monday morning" → **sarcasm** (true: anger/sadness, predicted: joy)

BERT sees "great" as a positive signal without pragmatic reasoning.

### 3. Label Ambiguity

Some emotions are inherently overlapping. The boundary between `love` and `joy`, or `anger` and `sadness`, is fuzzy even for human annotators.

### 4. Class Imbalance Residual Effects

Even with weighted loss, minority classes (`surprise`: 3.6%) may still underperform because:
- Fewer diverse examples → narrower learned distribution
- Weight amplification increases gradient variance for small classes

### 5. Domain Shift

The Emotion dataset is derived from Twitter. Applying the model to formal text (news, academic) may degrade performance due to domain mismatch.

---

## Class Imbalance Handling

### Strategy: Inverse-Frequency Weighted Cross-Entropy

```python
# data/dataset.py
def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    weights = [total / (num_classes * counts[c]) for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)
```

### Why Not Oversampling?

| Criterion | Weighted Loss | Oversampling |
|---|---|---|
| Training time | Unchanged | Increases proportionally |
| Overfitting risk | Low | High (duplicated samples) |
| Implementation | Loss function modification | Dataset modification |
| Gradient quality | Analytically controlled | Same noisy gradients, repeated |
| Compatibility | Native to PyTorch | Requires custom sampler |

### Expected Impact

| Metric | Without Weighting | With Weighting | Δ |
|---|---|---|---|
| Macro-F1 | ~0.87 | ~0.91 | **+4%** |
| Recall (surprise) | ~0.55 | ~0.72 | **+17%** |
| Recall (love) | ~0.70 | ~0.82 | **+12%** |
| Accuracy | ~0.93 | ~0.92 | -1% |

> **Key insight**: Weighted loss trades a small amount of majority-class accuracy for a significant gain in minority-class recall. This is the correct trade-off when all classes matter equally.

---

## Regularization

| Technique | Configuration | Purpose |
|---|---|---|
| **Weight decay** | 0.01 | L2 regularization on all parameters except biases and LayerNorm |
| **Dropout** | 0.1 (BERT default) | Stochastic regularization in attention and feed-forward layers |
| **Early stopping** | Patience = 3 epochs | Halt training when validation Macro-F1 stops improving |
| **Warmup** | 6% of total steps | Prevents large early gradient updates from destabilizing pre-trained weights |

### Early Stopping

```
Epoch 1: macro_f1 = 0.85  ✓ improving
Epoch 2: macro_f1 = 0.89  ✓ improving
Epoch 3: macro_f1 = 0.91  ✓ improving (best)
Epoch 4: macro_f1 = 0.90  ✗ patience 1/3
Epoch 5: macro_f1 = 0.90  ✗ patience 2/3
Epoch 6: macro_f1 = 0.89  ✗ patience 3/3 → STOP, revert to epoch 3
```

---

## Supervision Type

**Fully supervised, single-label, multi-class classification.**

| Property | Value |
|---|---|
| Supervision | Fully supervised (labeled train/val/test splits) |
| Task type | Multi-class classification (6 classes) |
| Label type | Single-label (each sample has exactly one emotion) |
| Loss function | Cross-entropy (with class weights) |
| Output layer | Linear head → softmax over K=6 classes |

No self-supervised, semi-supervised, or unsupervised components are used. The pre-trained BERT weights serve as the initialization, and all layers are fine-tuned end-to-end.

---

## Compute & Profiling

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| **GPU** | NVIDIA T4 (16 GB) | NVIDIA RTX 3090/4090 (24 GB) |
| **RAM** | 16 GB | 32 GB |
| **Disk** | 5 GB | 10 GB |
| **CUDA** | 12.1 | 12.1+ |

### Expected Training Profile

| Metric | T4 (Colab) | RTX 3090 | RTX 4090 |
|---|---|---|---|
| **Time per epoch** | ~3 min | ~1.5 min | ~1 min |
| **Total training** | ~15 min | ~8 min | ~5 min |
| **Peak GPU memory** | ~4.5 GB | ~4.5 GB | ~4.5 GB |
| **FP16 memory savings** | ~40% | ~40% | ~40% |

### Profiling Outputs

Training profile is automatically saved to `models/bert-emotion/training_profile.json`:

```json
{
  "training_time_seconds": 480.5,
  "training_time_minutes": 8.01,
  "peak_gpu_memory_gb": 4.52,
  "gpu_device": "NVIDIA GeForce RTX 3090",
  "epochs_completed": 3
}
```

### Memory Breakdown (Approximate)

| Component | Memory |
|---|---|
| BERT parameters (110M) | ~0.44 GB (FP16) |
| Optimizer states (Adam) | ~0.88 GB |
| Activations (batch=32, seq=128) | ~2.5 GB |
| Gradients | ~0.44 GB |
| **Total** | **~4.3 GB** |

---

## Reproducibility

### Seed Control

All sources of randomness are controlled:

```python
import random, numpy as np, torch, os

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)
```

### Remaining Non-Determinism

Even with all seeds set, **exact bit-level reproducibility across runs is not guaranteed** due to:

| Source | Cause | Impact |
|---|---|---|
| **CUDA atomic operations** | Floating-point addition is non-associative; kernel-level parallelism changes reduction order | ±0.001 in loss |
| **DataLoader workers** | Multi-process data loading introduces OS-level scheduling variance | Batch order may vary |
| **cuDNN algorithm selection** | Mitigated by `deterministic=True`, but some ops have no deterministic kernel | Rare; raises warning |
| **Cross-hardware** | Different GPU architectures use different floating-point implementations | Results differ across GPUs |

**Recommendation**: Report metrics as mean ± std over 3+ runs with different seeds for publication-quality results.

### Experiment Configuration

All hyperparameters are tracked in `config.yaml`. The full configuration is logged at the start of each training run for audit purposes.

---

## Bias & Ethical Considerations

### Model Bias

1. **Pre-training corpus bias**: BERT was trained on BooksCorpus and English Wikipedia, which encode societal biases present in those texts (gender, racial, cultural).

2. **Dataset bias**: The Emotion dataset is derived from English tweets, introducing:
   - **Demographic bias**: Twitter user demographics skew young, urban, English-speaking
   - **Temporal bias**: Language and emotional expression evolve over time
   - **Platform bias**: Tweet-style writing (abbreviations, hashtags) differs from general English

3. **Label bias**: Emotion annotation is subjective. Inter-annotator agreement is imperfect, and the label taxonomy (6 emotions) is a simplification of the full emotional spectrum (Ekman's model). Emotions like contempt, shame, or guilt are absent.

### Mitigation Strategies

- **Weighted loss** partially addresses label frequency bias but does not correct systematic annotation errors.
- **Error analysis** reveals which classes are systematically confused, surfacing potential annotation inconsistencies.
- Users should **not deploy this model** for high-stakes emotion detection (e.g., mental health screening, law enforcement) without extensive validation on the target population.

### Responsible Use

This project is intended for **research and educational purposes**. The model:
- Should not be used as a sole decision-maker for any consequential outcome
- May not generalize to languages other than English
- May produce harmful or incorrect predictions for sarcasm, irony, or culturally specific expressions

---

## Results

> **Note**: Results below are expected ranges based on the architecture and dataset. Run `python train.py` followed by `python evaluate.py` to generate actual numbers.

### Expected Performance

| Metric | Value (Expected) |
|---|---|
| **Accuracy** | 0.92 – 0.93 |
| **Macro-F1** | 0.89 – 0.92 |
| **Weighted-F1** | 0.92 – 0.93 |

### Expected Per-Class Performance (with Weighted Loss)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| sadness | 0.95 | 0.95 | 0.95 |
| joy | 0.96 | 0.96 | 0.96 |
| love | 0.82 | 0.84 | 0.83 |
| anger | 0.91 | 0.90 | 0.90 |
| fear | 0.89 | 0.90 | 0.89 |
| surprise | 0.72 | 0.75 | 0.73 |

---

## References

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**. *NAACL-HLT*. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

2. Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). **CARER: Contextualized Affect Representations for Emotion Recognition**. *EMNLP*. [Dataset](https://huggingface.co/datasets/dair-ai/emotion)

3. King, G., & Zeng, L. (2001). **Logistic Regression in Rare Events Data**. *Political Analysis*, 9(2), 137–163.

4. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). **Focal Loss for Dense Object Detection**. *ICCV*. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

5. Wolf, T., et al. (2020). **Transformers: State-of-the-Art Natural Language Processing**. *EMNLP (System Demonstrations)*. [arXiv:2003.10555](https://arxiv.org/abs/2003.10555)

---

## License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <i>Built with focus on reproducibility, rigor, and research clarity.</i>
</p>
