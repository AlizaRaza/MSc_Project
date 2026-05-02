# Explainable AI for Autism Spectrum Disorder Detection via Facial Image Analysis

**MSc Robotics and Artificial Intelligence — Queen Mary University of London**  
**Module: EMS715P | Supervised by Dr. M. Hasan Shaheed**

---

## Overview

This project investigates the use of **Explainable AI (XAI)** techniques for detecting Autism Spectrum Disorder (ASD) through facial image analysis. The core contribution is the development of attention-based Convolutional Neural Networks (CNNs) combined with **Grad-CAM heatmaps** to provide clinical interpretability — visualising *which* facial regions drive model decisions, addressing a key gap where most existing models operate as black boxes.

---

## Pipeline

The implementation is structured across four progressive stages:

### Stage 1 — VGG16 Baseline (`stage1_vgg16.py`)
- Model: VGG16 pretrained on ImageNet
- Dataset: FER-Autism (Mendeley Data)
- Task: Multi-class emotion classification
- Result: ~42% test accuracy — confirmed ImageNet domain gap on specialised medical data

### Stage 2 — Ensemble (`stage2_ensemble.py`)
- Model: VGG16 + Xception soft-voting ensemble
- Dataset: FER-Autism
- Result: ~34% — underperformed baseline due to correlated errors on small dataset; scientifically informative finding

### Stage 3 — Attention CNN with Grad-CAM (`stage3_attention.py`)
- Model: VGG16 + CBAM (Convolutional Block Attention Module)
- Dataset: FER-Autism
- Result: ~50% test accuracy; Grad-CAM heatmaps consistently highlighted eyes, nasal bridge, and perioral region — consistent with ASD phenotype literature

### Stage 4 — Binary ASD vs. TD Classifier (`stage4_binary.py`)
- Model: VGG16 + CBAM
- Dataset: FADC (Facial ASD Detection Challenge)
- Task: Binary classification — ASD vs. Typically Developing (TD)
- Result: ~99% test accuracy, AUC ~1.00; heatmaps revealed diffuse lower-face focus for ASD and intense eye-region focus for TD — directly consistent with ASD gaze research

---

## Key Findings

- Interpretability over accuracy: attention heatmaps with clinically grounded facial region focus are more valuable than marginal accuracy gains from black-box models on small medical datasets
- Negative results matter: the ensemble underperforming the baseline is a publishable, scientifically informative finding
- Transfer learning has fundamental limits on small, specialised medical imaging datasets

---

## Datasets

- **FER-Autism Dataset** — Mahmoud (2024), Mendeley Data
- **FADC Dataset** — GitHub

*Datasets are not included in this repository due to size and licensing.*

---

## Requirements

```bash
pip install tensorflow keras numpy matplotlib scikit-learn opencv-python
```

---

## Results

Result plots and Grad-CAM heatmaps for each stage are stored in the `results_stage*/` folders.

---

*Dissertation submitted in partial fulfilment of the MSc in Robotics and Artificial Intelligence, QMUL, 2026.*
