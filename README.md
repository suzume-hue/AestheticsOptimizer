# Aesthetics Optimizer

<img width="2880" height="1800" alt="image" src="https://github.com/user-attachments/assets/8aa9936f-0460-466d-a0ce-781537061d82" />

Personalised fMRI prediction from aesthetic image history.

Given a user's ranked history of image, this system predicts the whole-brain fMRI response to a new unseen image. The architecture uses cross-attention over calibration sets of image–fMRI pairs, encoding visual semantics and low-level perceptual structure to predict activation across 1,024 DiFuMo brain parcels (Yeo-7 network breakdown included). It then uses non-gradient based optimisation approaches to optimise the aesthetics of a generative flow field, through the limbic score.

Trained on the [BOLD5000](https://bold5000-dataset.github.io/website/) dataset via the [WAVE-BOLD5000](https://huggingface.co/datasets/PPWangyc/WAVE-BOLD5000) preprocessed release. Written up, including what failed — in the accompanying [Medium article](https://suzume1.medium.com/i-tried-to-build-an-aesthetic-model-of-the-human-brain-i-failed-heres-what-i-learned-13c75644258a).

---

## Repository structure

```
aesthetic-brain/
│
├── notebooks/
│   ├── notebook_a_preprocessing.py   # Feature extraction: MobileCLIP + MobileNetV3-Small
│   └── notebook_b_training.py        # Model training, evaluation, Yeo-7 breakdown
│
├── app/
│   ├── main.py                       # PyQt6 application entry point
│   ├── inference.py                  # Model inference pipeline
│   ├── calibration.py                # Calibration set management
│   ├── brain_viz.py                  # Yeo-7 network activation panel (animated)
│   ├── flowfield.py                  # Background visual
│   └── log_panel.py                  # Thread-safe real-time log panel
│
└── model/
    └── best_model.tar.xz             # Compressed trained model checkpoint
```

---

## Model

The trained checkpoint is compressed. To decompress:

```bash
cd model/
tar -xJvf best_model.tar.xz
```

This produces `best_model.pt`, which contains the model state dict and the training config (`CFG`).

**Architecture:** `CalibratedBoldPredictor` — cross-attention over calibration tokens (image features + Fourier-encoded aesthetic rank scores), feeding into a 4-layer MLP predicting 1,024-dimensional whole-brain BOLD activation.

**Training results:**

| Subject | Mean Pearson r (1024 parcels) |
|---------|-------------------------------|
| CSI1    | 0.1243                        |
| CSI2    | 0.0347                        |
| CSI3    | 0.0812                        |
| CSI4    | 0.0694                        |
| **Mean**| **0.0774**                    |

Visual network predictions were strongest (r ≈ 0.14–0.28). Limbic network — most relevant to aesthetic reward — did not recover meaningful signal.

---

## Notebooks

Run on Kaggle with GPU. Notebook A handles data download from HuggingFace, feature extraction (MobileCLIP-S0 + MobileNetV3-Small multi-layer hooks), and BOLD preprocessing. Notebook B handles training, evaluation, and Yeo-7 network breakdown.

You will need a HuggingFace token set as a Kaggle secret (`HF_TOKEN`) to download the WAVE-BOLD5000 dataset.

---

## Desktop application

The PyQt6 app (`app/main.py`) runs inference interactively, shows live Yeo-7 network activation bars, an animated aesthetic score, a sparkline of score history and evolving generative art towards higher aesthetics.

**Dependencies:**

```bash
pip install torch torchvision open-clip-torch PyQt6 numpy
```

**Run:**

```bash
python app/main.py
```

Ensure all models are loaded up from the UI logs.

---

## Dataset

- [BOLD5000](https://bold5000-dataset.github.io/website/) — Chang et al., 2019. fMRI while viewing 5,000 real-world images, four subjects.
- [WAVE-BOLD5000](https://huggingface.co/datasets/PPWangyc/WAVE-BOLD5000) — Wang et al., 2026. Whole-brain preprocessed version with DiFuMo-1024 parcellation.

---
