# 🛰️ MOPR Geospatial Hackathon: Semantic Segmentation from Drone Imagery

<p align="center">
  <img src="https://img.shields.io/badge/python-3.14+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Model-SegFormer-orange.svg" alt="Model">
</p>

## 🚀 Overview

This project builds a deep learning pipeline for **semantic segmentation** of drone orthophotos for the [SVAMITVA Scheme](https://svamitva.nic.in/). It targets building footprint extraction, rooftop-class understanding, road and waterbody mapping, and infrastructure-aware geospatial intelligence.

## ✨ Key Features

- **Two-stage Training:** Pretrain on LoveDA, then Train on target geospatial data
- **SegFormer Backbone:** Transformer-based segmentation with a practical training stack
- **Config-Driven Workflow:** Shared and task-specific settings under `config/`
- **Robust Evaluation:** Quantitative metrics plus qualitative visualization outputs
- **Reliable Checkpointing:** Resume-safe transitions with model head compatibility handling

## 🗂️ Project Structure

```text
├── pretrain.py                    # Pretrain entrypoint
├── train.py                       # Train entrypoint
├── evaluate.py                    # Evaluation + visualization entrypoint
├── model.py                       # SegFormer model definition
├── losses.py                      # Losses and segmentation metrics
├── utils.py                       # Device/logging/checkpoint helper utilities
├── model.pt                       # Shared checkpoint artifact
├── config/
│   ├── shared.py                  # Shared hyperparameters
│   ├── pretrain.py                # Pretrain-specific settings
│   ├── train.py                   # Train-specific settings
│   └── eval.py                    # Evaluation settings
├── processing/
│   ├── dataset.py                 # Dataset classes and loaders
│   ├── transforms.py              # Data transforms and augmentation
│   ├── preprocessing.py           # Input preprocessing helpers
│   └── postprocessing.py          # Mask post-processing helpers
├── training/
│   ├── train.py                   # Core training loop
│   ├── pretrain.py                # Pretrain-related training helpers
│   ├── phase_io.py                # Checkpoint/data IO helpers
│   └── primitives.py              # Shared training primitives
└── data/                          # Local dataset roots and demos
```

## ⚙️ Installation & Environment Setup

This repository uses [uv](https://github.com/astral-sh/uv) and `pyproject.toml` for dependency management.

```bash
uv sync
source .venv/bin/activate
```

## ⚙️ Configuration

Runtime behavior is controlled through files in `config/` (not CLI flags).

**Shared defaults** (`config/shared.py`):

- `LEARNING_RATE = 6e-5`
- `WEIGHT_DECAY = 0.01`
- `WARMUP_EPOCHS = 5`
- `BATCH_SIZE = 8`
- `NUM_WORKERS = 2`
- `VAL_INTERVAL = 1`
- `GRAD_ACCUM_STEPS = 1`
- `USE_GRADIENT_CHECKPOINTING = False`
- `MODEL_PATH = "model.pt"`

**Pretrain defaults** (`config/pretrain.py`):

- `NUM_CLASSES_PRETRAIN = 8`
- `NUM_EPOCHS_PRETRAIN = 20`
- `NUM_VAL_SAMPLES_PRETRAIN = 150`
- `PRETRAIN_DATA_ROOT = "./data/phase-2"`

**Train defaults** (`config/train.py`):

- `NUM_CLASSES_TRAIN = 4`
- `NUM_EPOCHS_TRAIN = NUM_EPOCHS_PRETRAIN + 50`
- `NUM_VAL_SAMPLES_TRAIN = 280`
- `TRAIN_IMG_DIR = "data/phase-3/TrainningDataset/processed_datasets"`
- `TRAIN_MASK_DIR = "data/phase-3/TrainningDataset/processed_masks"`
- `VAL_IMG_DIR = "data/phase-3/ValidationDataset/processed_datasets"`
- `VAL_MASK_DIR = "data/phase-3/ValidationDataset/processed_masks"`

## 💾 Data Setup

**Pretrain data (LoveDA)**

- Expected root: `data/phase-2/`
- Scenes configured by default: `rural`, `urban`

**Train data (target geospatial dataset)**

- Train images: `data/phase-3/TrainningDataset/processed_datasets`
- Train masks: `data/phase-3/TrainningDataset/processed_masks`
- Val images: `data/phase-3/ValidationDataset/processed_datasets`
- Val masks: `data/phase-3/ValidationDataset/processed_masks`

**Mask convention**

- Label value `255` is treated as ignore region for VOC-style masks.

## 🏋️ Training & Evaluation Workflow

Run in this order for best results:

```bash
# 1) Pretrain
uv run pretrain.py

# 2) Train
uv run train.py

# 3) Evaluate
uv run evaluate.py
```

Equivalent VS Code tasks are available: `Pretrain`, `Train`, and `Evaluate`.

## 🧠 Model & Training Details

- **Architecture:** [SegFormer](https://arxiv.org/abs/2105.15203) (MiT-b2 via `segmentation_models_pytorch`)
- **Primary metrics:** mIoU, per-class IoU, pixel accuracy
- **Evaluation script outputs:** Mean Pixel Accuracy, Mean IoU, and processed-mask variants
- **Checkpoints:** Pretrain/Train flows read and write `model.pt`
- **Resume behavior:** On class-count mismatch, incompatible segmentation head state is dropped to allow clean continuation

## 📊 Results & Visualizations

`evaluate.py` computes metrics and displays side-by-side plots for input, ground truth, prediction, and processed prediction.

## 🙏 Acknowledgements

- [SVAMITVA Scheme](https://svamitva.nic.in/)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [LoveDA Dataset](https://github.com/Junjue-Wang/LoveDA)
- [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
