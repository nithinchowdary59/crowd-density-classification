# Crowd Density Image Classification Framework

A deep learning system that classifies crowd images into four density levels using CNN and transfer learning models. Built for public safety applications including stampede and crowd-crush early warning.

---

## Project Overview

| Item | Details |
|------|---------|
| Task | 4-class image classification |
| Classes | Low / Medium / High / Very High |
| Best Model | EfficientNetB0 — 83.27% accuracy |
| Dataset Size | 7,558 images (merged, balanced) |
| Environment | Kaggle Notebook — GPU T4 x2 |

---

## Datasets

| Dataset | Link | Images Used | Label Method |
|---------|------|-------------|--------------|
| ShanghaiTech | [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech) | 1,198 | .mat head count annotations |
| JHU-Crowd++ | [Kaggle](https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd) | 4,360 | YOLO .txt head count annotations |
| Abnormal High-Density Crowds | [Kaggle](https://www.kaggle.com/datasets/angelchi56/abnormal-highdensity-crowds) | 2,000 | Scene-based (all Very High) |

**Class distribution after merging:**

| Class | Count | Percentage |
|-------|-------|------------|
| Low | 1,869 | 24.7% |
| Medium | 1,842 | 24.4% |
| High | 1,847 | 24.4% |
| Very High | 2,000 | 26.5% |
| **Total** | **7,558** | |

---

## Models and Results

| Model | Test Accuracy | Precision | Recall | F1-Score | Inference Time |
|-------|--------------|-----------|--------|----------|----------------|
| Custom CNN (from scratch) | 82.74% | 83.01% | 82.40% | 82.51% | 6.61 ms/img |
| ResNet50 (Transfer Learning) | 81.48% | 80.93% | 81.07% | 80.91% | 4.76 ms/img |
| EfficientNetB0 (Transfer Learning) | **83.27%** | **83.44%** | **82.93%** | **83.08%** | **3.84 ms/img** |

**EfficientNetB0 achieves the best accuracy (83.27%) with the fastest inference time (3.84 ms/image).**

---

## Per-Class Results (Best Model — EfficientNetB0)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Low | 0.85 | 0.77 | 0.81 | 374 |
| Medium | 0.66 | 0.73 | 0.69 | 368 |
| High | 0.83 | 0.82 | 0.83 | 370 |
| Very High | 1.00 | 1.00 | 1.00 | 400 |

---

## How to Run

### On Kaggle (Recommended)

1. Create a new Kaggle notebook
2. Add these 3 datasets as input:
   - `tthien/shanghaitech`
   - `hoangxuanviet/jhu-crowd`
   - `angelchi56/abnormal-highdensity-crowds`
3. Set accelerator to **GPU T4 x2**
4. Upload and import the notebook file
5. Click **Run All**

### Requirements

```
tensorflow >= 2.19.0
numpy
pandas
opencv-python
matplotlib
seaborn
scikit-learn
scipy
```

---

## Key Technical Decisions

**Why tf.data pipeline instead of loading all images into RAM?**
Loading 7,558 images at 224×224×3 float32 would require ~10GB RAM, exceeding Kaggle's 13GB limit during training. The tf.data pipeline streams images batch-by-batch, keeping RAM usage under 2GB.

**Why tercile thresholds instead of fixed values?**
Data-driven tercile thresholds (T1=66, T2=208 head counts) computed from the actual ShanghaiTech + JHU distribution ensure equal class sizes for Low/Medium/High, avoiding arbitrary label boundaries.

**Why single GPU instead of MirroredStrategy?**
MirroredStrategy caused validation loss corruption (val_loss > 10) due to mixed JPG/PNG formats in the dataset across GPUs. Single GPU training produced stable, correct validation metrics.

**Why 128×128 for Custom CNN?**
Reducing input resolution from 224×224 to 128×128 for the from-scratch CNN reduces overfitting risk and speeds up convergence, while pretrained transfer learning models retain 224×224 to preserve ImageNet feature quality.

---

## Project Structure

```
crowd-density-classification/
│
├── crowd-density-final-code.ipynb    # Main Kaggle notebook
├── README.md                         # This file
├── dataset_links.md                  # All dataset links
│
└── figures/                          # Output figures (PDF)
    ├── class_distribution.pdf
    ├── sample_images.pdf
    ├── augmentation_examples.pdf
    ├── cnn_training_curves.pdf
    ├── cnn_confusion_matrix.pdf
    ├── cnn_roc.pdf
    ├── resnet50_training_curves.pdf
    ├── resnet50_confusion_matrix.pdf
    ├── resnet50_roc.pdf
    ├── efficientnet_training_curves.pdf
    ├── efficientnet_confusion_matrix.pdf
    ├── efficientnet_roc.pdf
    ├── model_comparison.pdf
    ├── gradcam_visualizations.pdf
    └── error_analysis.pdf
```

---

## Research Questions Addressed

- **RQ1:** How accurately can CNN-based models classify crowd density? → **83.27% (EfficientNetB0), 82.74% (Custom CNN), 81.48% (ResNet50)**
- **RQ2:** Which model provides the best accuracy-efficiency balance? → **EfficientNetB0 — highest accuracy AND fastest inference (3.84 ms/img)**
- **RQ3:** How do preprocessing and class balancing affect performance? → **Tercile thresholds + tf.data augmentation + class weights improved CNN from 56% to 82.74%**
- **RQ4:** Do Grad-CAM maps confirm meaningful crowd region attention? → **Yes — heatmaps consistently focus on crowd-dense image regions**
- **RQ5:** What are the main failure patterns? → **Medium-High boundary confusion is primary failure; Very High misclassification is zero across all models**

---

## Kaggle Notebook

🔗 [View Notebook](#) *https://www.kaggle.com/code/nithinchowdaryraya/crowd-density-final-code*

---

*ML Project — Phase 2 | Crowd Density Classification Framework | Idea 50 of 75*
