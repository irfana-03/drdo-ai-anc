# DRDO AI-ANC

**AI-Driven Context-Aware Adaptive Noise Cancellation for Mission-Critical Communication**

> Research prototype for SIH 2026 — Suppress stationary, non-stationary, and impulsive acoustic noise while preserving speech intelligibility, with real-time edge-hardware inference.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Acquisition](#dataset-acquisition)
- [Dataset Licenses](#dataset-licenses)
- [Verifying the Installation](#verifying-the-installation)
- [Usage](#usage)
- [Design Principles](#design-principles)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DRDO AI-ANC Pipeline                  │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Data    │  Pre-    │  Noise   │  Speech  │  Real-Time  │
│  Ingest  │  process │  Classify│  Enhance │  Inference  │
│          │          │          │          │             │
│ CHiME-3  │ Resample │ Classify │DeepFilter│ Live mic    │
│ DEMAND   │ Mono     │ noise    │ Net /    │ block-based │
│ SONYC-UST│ Normalise│ type     │ RNNoise  │ streaming   │
│          │ Features │          │          │             │
└──────────┴──────────┴──────────┴──────────┴─────────────┘
                         │
                    ┌────┴────┐
                    │Evaluate │
                    │ PESQ    │
                    │ STOI    │
                    │ SI-SDR  │
                    └─────────┘
```

### Pipeline Components

| Stage              | Module                           | Status     |
|--------------------|----------------------------------|------------|
| Data management    | `src/data/dataset_manager.py`    | ✅ Done    |
| Audio I/O          | `src/data/audio_loader.py`       | ✅ Done    |
| Dataset validation | `scripts/verify_real_dataset.py` | ✅ Done    |
| Preprocessing      | `src/preprocessing/audio_preprocessing.py` | ✅ Done |
| Feature extraction | `src/preprocessing/feature_extraction.py`  | ✅ Done |
| Noise classification | `src/classification/noise_classifier.py` | ✅ Done |
| Impulse detection  | `src/classification/impulse_detector.py` | ✅ Done |
| Evaluation metrics | `src/evaluation/classification_metrics.py` | ✅ Done |
| Speech enhancement | `src/enhancement/`               | 🔜 Planned |
| Adaptive control   | `src/adaptive/`                  | 🔜 Planned |
| Real-time engine   | `src/realtime/`                  | 🔜 Planned |

---

## Project Structure

```
drdo_ai_anc/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git exclusions
├── config/
│   └── config.yaml                     # Master configuration
├── data/
│   ├── raw/                            # Original recordings (never modified)
│   │   ├── chime3/                     # CHiME-3 dataset
│   │   ├── demand/                     # DEMAND dataset
│   │   └── sonyc_ust/                  # SONYC-UST dataset
│   ├── processed/                      # Preprocessed audio
│   ├── metadata/                       # Auto-generated metadata CSVs
│   └── README.md                       # Data acquisition guide
├── models/
│   ├── pretrained/                     # Downloaded model weights
│   └── custom/                         # Trained / fine-tuned models
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── __main__.py                 # CLI entry point
│   │   ├── dataset_manager.py          # Dataset registry and CLI
│   │   ├── audio_loader.py             # Audio I/O utilities
│   │   └── dataset_validator.py        # Validation & metadata generation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── audio_preprocessing.py      # Resample, mono, normalise
│   │   └── feature_extraction.py       # STFT, mel, RMS, ZCR, …
│   ├── enhancement/                    # (planned) DeepFilterNet, RNNoise
│   ├── classification/                 # (planned) Noise-type classifier
│   ├── adaptive/                       # (planned) Context-aware adaptation
│   ├── realtime/                       # (planned) Live microphone inference
│   └── evaluation/                     # (planned) PESQ, STOI, SI-SDR
├── scripts/
│   ├── setup_environment.py            # Dependency checker
│   ├── download_datasets.py            # Download real datasets from Zenodo
│   ├── verify_real_dataset.py          # Validate audio + build metadata/splits
│   ├── dataset_report.py               # Print dataset statistics
│   ├── train_classifier.py             # Train noise context classifier
│   ├── evaluate_classifier.py          # Evaluate on held-out test set
│   └── verify_installation.py          # Full project verification
├── train.py                            # End-to-end pipeline runner
├── tests/
│   ├── __init__.py
│   └── test_audio_load.py              # Load one real audio file and inspect
└── app/
    └── __init__.py                     # Application entry point (planned)
```

---

## Installation

### Prerequisites

- **Python ≥ 3.9** (3.10+ recommended)
- **pip** or **conda** package manager
- (Optional) NVIDIA GPU with CUDA for accelerated inference

### Step 1 — Clone and enter the project

```bash
git clone <your-repo-url>
cd drdo_ai_anc
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** If you need GPU support, install PyTorch with the
> appropriate CUDA version first:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```
> Then install the remaining requirements.

### Step 4 — Verify the installation

```bash
python scripts/setup_environment.py
python scripts/verify_installation.py
```

---

## Dataset Acquisition

All datasets used in this project are **real recorded data** — no synthetic
audio is generated for evaluation.

### Quick Reference

| Dataset    | Type                | Source                                                   | Auto-download? |
|------------|---------------------|----------------------------------------------------------|----------------|
| CHiME-3    | Noisy speech        | [chimechallenge.org](https://www.chimechallenge.org/challenges/chime3) | ❌ Requires registration |
| DEMAND     | Environmental noise | [Zenodo #1227121](https://zenodo.org/record/1227121)     | ✅ Free (CC BY-SA 4.0) |
| SONYC-UST  | Urban acoustics     | [Zenodo #3966543](https://zenodo.org/record/3966543)     | ✅ Free (CC BY 4.0) |

### CHiME-3

1. Visit the [CHiME-3 challenge page](https://www.chimechallenge.org/challenges/chime3).
2. Register and accept the license agreement.
3. Download and extract into `data/raw/chime3/`.

### DEMAND

1. Download from [Zenodo](https://zenodo.org/record/1227121).
2. Extract into `data/raw/demand/`.

### SONYC-UST

1. Download from [Zenodo](https://zenodo.org/record/3966543).
2. Extract into `data/raw/sonyc_ust/`.

Run the dataset helper to check status:

```bash
python scripts/download_datasets.py --check
```

---

## Dataset Licenses

| Dataset   | License                | Redistribution | Commercial Use |
|-----------|------------------------|----------------|----------------|
| CHiME-3   | Academic (restricted)  | ❌ No          | ❌ No          |
| DEMAND    | CC BY-SA 4.0          | ✅ Yes (share-alike) | ✅ Yes   |
| SONYC-UST | CC BY 4.0             | ✅ Yes         | ✅ Yes         |

---

## Verifying the Installation

Run the comprehensive verification:

```bash
python scripts/verify_installation.py
```

This checks:
- Python version
- All required package imports (PyTorch, torchaudio, numpy, scipy, librosa,
  soundfile, scikit-learn, sounddevice, PyYAML, matplotlib, pandas)
- Project directory structure
- Configuration file loading
- Module imports
- Dataset manager initialisation

---

## Usage

### Full ML Pipeline (recommended)

```bash
python train.py
```

### Presentation Demo (fastest)

```bash
run_demo.bat
```

Or manually:

```bash
streamlit run app/dashboard.py
```

**Offline demo:** On the Live Monitor page, upload a WAV file or use the built-in sample, then click **PROCESS OFFLINE FILE**.

**SIH Demo Mode:** Click **LAUNCH SIH DEMO MODE** in the sidebar for a clean presentation view.

```bash
python scripts/download_datasets.py      # Download DEMAND + SONYC-UST subset
python scripts/verify_real_dataset.py    # Validate + build metadata/splits
python scripts/dataset_report.py         # Print dataset statistics
python scripts/train_classifier.py       # Train CNN classifier
python scripts/evaluate_classifier.py    # Evaluate on held-out test set
```

### Datasets Actually Used (First Training Run)

| Dataset   | Recordings | Duration | Source | License |
|-----------|-----------|----------|--------|---------|
| DEMAND    | 4 (ch01 per environment) | 0.33 h | Zenodo #1227121 | CC BY-SA 4.0 |
| SONYC-UST | 506 | 1.41 h | Zenodo #3966543 (`audio-18.tar.gz`) | CC BY 4.0 |
| CHiME-3   | 0 (manual download required) | — | chimechallenge.org | Academic |

**Total: 510 real recordings, ~1.74 hours of audio. No synthetic data.**

DEMAND subset: `NRIVER_16k`, `TBUS_16k`, `OOFFICE_16k`, `STRAFFIC_16k` (channel 1 only).

### Label Mapping

Transparent mappings are in `data/metadata/label_mapping.csv` and `src/data/label_mapping.py`.

| Original Label | Mapped Context | Dataset |
|----------------|----------------|---------|
| NRIVER, OOFFICE | STATIONARY | DEMAND |
| TBUS, STRAFFIC | DYNAMIC | DEMAND |
| engine, powered-saw | DYNAMIC | SONYC-UST |
| machinery-impact, non-machinery-impact, alert-signal | IMPULSIVE | SONYC-UST |
| human-voice | SPEECH | SONYC-UST |
| music, dog | OTHER | SONYC-UST |

### Train/Validation/Test Split

- **Strategy:** Recording-level (no segment leakage)
- **SONYC-UST:** Uses original dataset splits (sensor-disjoint train/val, time-disjoint test)
- **DEMAND:** Hash-based split across environments
- **Ratios:** 70% / 15% / 15% (configurable in `config/config.yaml`)
- **Actual split:** 358 train / 129 validation / 23 test

### Feature Extraction

Log-mel spectrogram (64 mels, n_fft=2048, hop=512) at 16 kHz, plus RMS, spectral centroid, spectral flux, and zero-crossing rate. Parameters in `config/config.yaml`.

### Model Architecture

Compact CNN (`NoiseContextCNN`) on log-mel input:
- 3 conv blocks (16→32→64 channels) with BatchNorm + MaxPool
- AdaptiveAvgPool → FC(128) → Dropout(0.3) → FC(5 classes)
- Suitable for edge deployment (~50K parameters)

### Trained Model Location

```
models/custom/noise_context_classifier.pt
models/custom/class_mapping.json
models/custom/training_config.json
```

### Evaluation Results (Held-Out Test Set)

Results saved to `results/metrics/metrics.json` and `results/figures/confusion_matrix.png`.

| Metric | Value |
|--------|-------|
| Test accuracy | 0.435 |
| Macro F1 | 0.236 |

### Limitations

1. **CHiME-3 not included** — requires manual registration; add to `data/raw/chime3/` for speech evaluation data.
2. **Small first-run subset** — 4 DEMAND environments + 1 SONYC archive (~700 MB compressed).
3. **Imbalanced classes** — OTHER and DYNAMIC dominate; class weighting applied but performance varies.
4. **Small test set** — 23 recordings (SONYC original test split intersecting downloaded archive).
5. **No synthetic data** — all training uses verified real recordings only.

### Dataset Manager CLI

```bash
# List all registered datasets
python -m src.data.dataset_manager --list

# Validate a dataset
python -m src.data.dataset_manager --validate demand

# Generate metadata CSV
python -m src.data.dataset_manager --metadata demand

# Preprocess a dataset (resample, mono, normalise)
python -m src.data.dataset_manager --prepare demand

# Split a dataset (recording-level, no data leakage)
python -m src.data.dataset_manager --split demand
```

### Test Audio Loading

```bash
# Auto-discover a file in data/raw/
python tests/test_audio_load.py

# Or specify a file explicitly
python tests/test_audio_load.py data/raw/demand/DKITCHEN/ch01.wav
```

---

## Design Principles

1. **Real data only** — no synthetic datasets for evaluation.
2. **Reproducibility** — seeded splits, deterministic pipelines, version-pinned
   dependencies.
3. **Data safety** — raw recordings are never overwritten; processed outputs go
   to separate directories.
4. **Modularity** — every pipeline stage is a self-contained module that can be
   replaced or extended independently.
5. **Data-leakage prevention** — dataset splits are performed at the
   recording / source level, not at the segment level.
6. **Edge-ready** — the architecture targets real-time inference on
   resource-constrained hardware.

---

## Primary Models

| Model          | Repository                                        | Role      |
|----------------|---------------------------------------------------|-----------|
| DeepFilterNet  | https://github.com/Rikorose/DeepFilterNet         | Primary   |
| RNNoise        | https://github.com/xiph/rnnoise                   | Baseline  |

---

## License

This project is a research prototype developed for SIH 2026.
Individual dataset licenses are documented above.
