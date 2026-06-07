# Crowd Density Image Classification Framework
### A CNN-Based Decision-Support System for Public Safety Monitoring
**Machine Learning Project — Phase 2 | Idea 50 of 75**

---

## Project Overview

This project develops a **CNN-based Crowd Density Image Classification Framework** that automatically classifies crowd images into four density levels:

| Class | Label | Head Count Threshold |
|-------|-------|---------------------|
| Low | 0 | Count ≤ 80 |
| Medium | 1 | 80 < Count ≤ 163 |
| High | 2 | 163 < Count ≤ 311 |
| Very High | 3 | Count > 311 |

The system is designed as a **decision-support tool** that assists security personnel in detecting dangerously high crowd densities before situations become life-threatening — it assists human experts rather than replacing them.

---

## Research Questions

This project addresses the following publication-ready research questions:

**RQ1:** How accurately can CNN-based models perform crowd density classification from images using publicly available image datasets?

**RQ2:** Which model family — custom CNN or transfer learning — provides the best balance between predictive performance and computational efficiency for this task?

**RQ3:** How do preprocessing, augmentation, and class-imbalance handling influence model robustness and class-wise performance?

**RQ4:** Do Grad-CAM explanations show that the trained models focus on meaningful visual regions related to crowd density classification from images?

**RQ5:** What are the main failure patterns, deployment limitations, and practical risks when applying this system in real security and surveillance settings?

---

## Datasets Used

| Dataset | Source | Images Used | Label Method |
|---------|--------|-------------|--------------|
| ShanghaiTech | [Kaggle](https://www.kaggle.com/datasets/tthien/shanghaitech) | 1,198 | .mat head count |
| JHU-Crowd++ | [Kaggle](https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd) | 3,000 | YOLO .txt head count |
| Abnormal High-Density Crowds | [Kaggle](https://www.kaggle.com/datasets/angelchi56/abnormal-highdensity-crowds) | 3,000 | Scene-based (Very High) |
| **Total** | | **7,198** | |

---

## Models Implemented

### 1. Custom CNN (Baseline)
- 5-block VGG-style architecture with BatchNormalization and Dropout
- Trained from scratch on the merged dataset
- Optimizer: AdamW with label smoothing
- **Test Accuracy: 56.11%**

### 2. EfficientNetV2S (Transfer Learning)
- Pretrained on ImageNet
- Two-phase training: frozen base → fine-tuning top 60 layers
- **Test Accuracy: 45.49%**

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Custom CNN | 56.11% | 29.74% | 40.61% | 30.00% |
| Custom CNN + TTA | 56.60% | 27.46% | 38.70% | 29.15% |
| EfficientNetV2S | 45.49% | 29.96% | 30.31% | 29.13% |
| EfficientNetV2S + TTA | 54.10% | 31.17% | 28.68% | 26.18% |

### ROC AUC Scores

| Class | Custom CNN | EfficientNetV2S |
|-------|------------|-----------------|
| Low | 0.691 | 0.596 |
| Medium | 0.695 | 0.449 |
| High | 0.483 | 0.594 |
| Very High | 0.857 | 0.623 |
| **Mean AUC** | **0.682** | **0.565** |

> **Note:** These results represent the current Phase 2 baseline. The model will be further improved in the final submission phase with better class balancing, extended fine-tuning, and architectural improvements.

---

## Project Structure

```
crowd-density-classification/
│
├── crowd-model-v2-clean.ipynb    # Main Kaggle notebook
├── README.md                     # This file
│
├── figures/                      # Output figures
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── cnn_training_curves.png
│   ├── efficientnet_training_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── efficientnet_confusion_matrix.png
│   ├── cnn_roc_curves.png
│   ├── efficientnet_roc_curves.png
│   ├── model_comparison.png
│   ├── gradcam_visualizations.png
│   └── error_analysis.png
│
└── proposal/                     # Phase 2 proposal document
    └── proposal.pdf
```

---

## How to Run

### On Kaggle (Recommended)
1. Open the notebook on Kaggle
2. Add the following datasets as input:
   - `tthien/shanghaitech`
   - `hoangxuanviet/jhu-crowd`
   - `angelchi56/abnormal-highdensity-crowds`
3. Set accelerator to **GPU T4 x2**
4. Click **Run All**

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

## Implementation Details

| Component | Details |
|-----------|---------|
| Image Size | 224 × 224 px |
| Batch Size | 32 |
| Max Epochs | 60 (CNN) / 40 (Transfer Learning) |
| Optimizer | AdamW |
| Loss | Categorical Cross-Entropy + Label Smoothing |
| Augmentation | Rotation, Flip, Zoom, Brightness, Shear |
| Class Balancing | Computed class weights via sklearn |
| Explainability | Grad-CAM visualizations |
| Environment | Kaggle Notebook — GPU T4 x2 |

---

## Key Features

-  Multi-dataset merging pipeline (3 datasets, 7,198 images)
-  Custom CNN trained from scratch
-  EfficientNetV2S transfer learning with two-phase fine-tuning
-  Data augmentation and class imbalance handling
-  Confusion matrix, ROC curves, classification report
-  Grad-CAM explainability visualizations
-  Test Time Augmentation (TTA)
-  Error analysis of misclassified samples
-  Model comparison table

---



## Limitations & Ethical Considerations

- Model trained on static images may not generalize to all camera angles or lighting
- Class boundary ambiguity between adjacent density levels
- Training dataset size (~7,000) is small for production-scale deployment
- Privacy concerns around automated crowd surveillance
- Bias risk if training data does not represent all demographic groups

---



## Kaggle Notebook

 [View Kaggle Notebook](#) *(https://www.kaggle.com/code/nithinchowdaryraya/nithin-crowd-density)*

---

*Machine Learning Project — Phase 2 Submission*
*Crowd Density Image Classification Framework *
