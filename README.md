# DeepBlood: Blood Cell Detection with YOLOv3

> Applying modern object detection architectures to high-precision analysis of blood smear images — detecting RBC, WBC, and Platelets using a from-scratch YOLOv3 implementation in PyTorch.

---

## Overview

DeepBlood is an end-to-end object detection pipeline for automated blood cell analysis. The project implements **YOLOv3 from scratch** — including the Darknet53 backbone, FPN neck, multi-scale prediction heads, custom loss functions, NMS, and anchor optimization via K-Means — and evaluates it on the Blood Cell Detection Dataset (Kaggle).

**Why it matters:** Accurate differential cell counting (RBC/WBC/Platelets ratio) is a key diagnostic marker. Automating this eliminates inter-laboratory variability, catches early-stage anomalies, and frees clinicians from routine slide review.

---

## Architecture

YOLOv3 is built entirely from scratch with the following components:

```
Darknet53 backbone
    ├── CNNBlock (Conv2d + BatchNorm + LeakyReLU)
    ├── ResidualBlock (skip connections)
    └── 5 stages → feature maps at P3, P4, P5 scales

YOLONeck_FPN
    └── Feature Pyramid Network with upsampling for multi-scale fusion

YOLOHeads (3 prediction scales: 13×13, 26×26, 52×52)
    └── PredictionHead per scale → 3 × (5 + num_classes) channels
```

**Loss functions (custom):**
- `BoxLoss` — MSE on bounding box center (cx, cy) and size (w, h)
- `ObjectnessLoss` — BCE for object presence/absence
- `ClassificationLoss` — BCE per class

---

## Dataset

**[Blood Cell Detection Dataset](https://www.kaggle.com/datasets/adhoppin/blood-cell-detection-datatset)** (Kaggle)

| Class | Description |
|---|---|
| `WBC` | White blood cells (leukocytes) |
| `RBC` | Red blood cells (erythrocytes) |
| `Platelets` | Thrombocytes |

Images: 416×416 px · YOLO format labels (normalized cx, cy, w, h)

---

## Pipeline

1. **Anchor optimization** — K-Means clustering on training bounding boxes to find 9 optimal anchors grouped by scale
2. **Training** — YOLOv3 on train split with ImageNet-normalized inputs
3. **Threshold tuning** — Grid search over confidence & IoU thresholds on validation set, optimized for mAP@.50:.95
4. **Evaluation** — mAP@.50:.95, mAP@.50, mAP@.75 on test set via `torchmetrics`
5. **Visualization** — Bounding box overlay with per-class color coding

---

## Results

| Metric | Score |
|---|---|
| **mAP@.50:.95** | **0.4087** |
| **mAP@.50** | **0.7770** |
| **mAP@.75** | 0.3656 |
| Inference time | 0.264 sec/image |
| FPS | ~3.8 |

**Key observations:**
- The model detects all three cell types effectively at IoU ≥ 0.50
- WBC false positives in background regions — main area for improvement
- mAP@.75 suggests room to improve bounding box localization precision
- ~4 FPS is suitable for offline analysis; YOLOv8-nano would be needed for real-time deployment

---

## Stack

```
Python              3.x
PyTorch             torch, torchvision
torchmetrics        MeanAveragePrecision
scikit-learn        KMeans (anchor optimization)
matplotlib / PIL    visualization
Dataset             Blood Cell Detection (Kaggle, YOLO format)
Platform            Google Colab (CUDA)
```

---

## How to Run

**1. Install dependencies**
```bash
pip install torch torchvision torchmetrics scikit-learn matplotlib Pillow tqdm
```

**2. Download dataset**
```bash
kaggle datasets download -d adhoppin/blood-cell-detection-datatset
```

**3. Run the notebook**
```bash
jupyter notebook DeepBlood.ipynb
```

Checkpoints are saved as `checkpoint_yolov3.pth` and reloaded for threshold tuning and evaluation.

---

## Project Structure

```
.
├── DeepBlood.ipynb
├── checkpoint_yolov3.pth         # saved model weights
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   └── test/
└── README.md
```

---

## Future Work

- Replace custom YOLOv3 with **YOLOv8** for improved accuracy and real-time FPS
- Add **data augmentation** (mosaic, mixup) to reduce WBC false positives
- Adapt pipeline for other biomarkers — malaria parasite detection, platelet aggregation, etc.

---

*Author: Anastasia Alexandrova · MIPT · 2025*
