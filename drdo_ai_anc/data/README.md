# Data Directory

This directory holds all audio datasets used by the DRDO AI-ANC project.

## Structure

```
data/
├── raw/            # Original, unmodified recordings (NEVER overwritten)
│   ├── chime3/     # CHiME-3 real noisy speech
│   ├── demand/     # DEMAND environmental noise
│   └── sonyc_ust/  # SONYC-UST urban sound recordings
├── processed/      # Pre-processed audio (resampled, normalised, mono)
├── metadata/       # Auto-generated CSV metadata per dataset
└── README.md       # This file
```

## Dataset Acquisition

| Dataset    | URL                                           | License              |
|------------|-----------------------------------------------|----------------------|
| CHiME-3    | https://www.chimechallenge.org/challenges/chime3 | Academic (restricted, registration required) |
| DEMAND     | https://zenodo.org/record/1227121             | CC BY-SA 4.0         |
| SONYC-UST  | https://zenodo.org/record/3966543             | CC BY 4.0            |

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

## Important Rules

- **Never** modify or overwrite files in `raw/`.
- All preprocessing outputs go to `processed/`.
- Metadata CSVs are generated automatically via `dataset_validator.py`.
