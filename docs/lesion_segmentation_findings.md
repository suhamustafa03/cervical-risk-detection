# Lesion Segmentation & Data Audit Findings

**Project:** Early Cervical Risk Detection using Explainable Colposcopy Analysis  
**Date:** October 2026  
**Document Status:** Complete Audit Report  

---

## 1. Executive Summary

This audit investigates the availability of lesion segmentation data, ground-truth masks, and segmentation models in the `cervical_project` codebase.

**Key Finding:**  
There is **no dedicated lesion segmentation model**, **no trained segmentation architecture** (such as U-Net, DeepLabV3+, or Mask R-CNN), and **no pixel-level lesion ground-truth masks** present in this repository.

To comply with clinical research integrity and ethical AI standards:
1. **No synthetic or fabricated lesion masks** have been generated.
2. The prototype strictly relies on **Grad-CAM (Gradient-weighted Class Activation Mapping)** for visual interpretability of the classification decision.
3. The UI and generated clinical reports display an explicit lesion limitation disclosure:
   > *"Dedicated lesion detection is not available in this prototype. A clean-looking image or the absence of highlighted regions does not confirm that the cervix is lesion-free."*

---

## 2. Codebase & Model Architecture Audit

### 2.1 Model Checkpoints Present

1. `phase2_resnet18_best.pth` (44.8 MB):
   - **Architecture:** Torchvision ResNet-18 with modified final fully-connected layer (`Linear(512, 3)`).
   - **Task:** 3-class classification of the Transformation Zone (Type 1, Type 2, Type 3).
   - **Training Source:** `data/phase2_split.csv` containing image paths and `patient_tz_type` labels.
   - **Output:** Categorical probability distribution over Transformation Zone types.

2. `cervical_validator.pth` (44.8 MB):
   - **Architecture:** Torchvision ResNet-18 with modified final layer (`Linear(512, 2)`).
   - **Task:** Binary input screening (cervical colposcopy vs. non-cervical image).
   - **Output:** Screening confidence score to gate invalid uploads.

Neither model contains decoder branches, segmentation heads, or pixel-wise prediction capabilities.

---

## 3. Dataset Audit

1. **`dataset/train` & `dataset/test`**:
   - Organized into directory classification categories: `CIN1`, `CIN2`, `CIN3`.
   - Contains raw JPEG/PNG image files only.
   - Contains zero `.mat`, `.npy`, `.png` binary mask annotations, COCO JSON polygon annotations, or Pascal VOC XML segmentations.

2. **`data/phase2_split.csv`**:
   - Contains metadata columns: `image_name`, `patient_id`, `patient_tz_type`.
   - No bounding box coordinates or contour points are present.

3. **`storage/`**:
   - Subdirectories: `originals`, `heatmaps`, `overlays`, `reports`, `masks`.
   - The `masks/` directory contains no pre-existing ground-truth masks.
   - In `backend_addons.py`, the `metrics_json` field explicitly records:
     ```json
     {
       "status": "not_computed",
       "reason": "No validated cervix/lesion segmentation mask is currently available."
     }
     ```

---

## 4. Grad-CAM vs. Lesion Segmentation

Grad-CAM highlights regions of the input image that exerted the strongest gradient influence on the model's final classification layer (`model.layer4[-1].conv2`).

**Critical Clinical Distinction:**
- **Grad-CAM is an attribution map**, not a lesion boundary.
- An intense Grad-CAM activation indicates where the network looked to classify the transformation zone (e.g. squamocolumnar junction, cervical os, vascular patterns).
- Grad-CAM **must not** be represented to clinicians as lesion margins, lesion depth, or biopsy guidance boundaries.

---

## 5. Recommended Requirements for Future Lesion Segmentation Implementation

If a true lesion segmentation pipeline is developed in future phases, the following must be provided:

1. **Expert-Annotated Dataset**:
   - Multi-observer manual delineations by certified colposcopists.
   - Inter-observer agreement metrics (Fleiss' Kappa, Dice similarity coefficient).
2. **Dedicated Segmentation Architecture**:
   - Fully convolutional network (e.g., U-Net, nnU-Net, SegFormer).
   - Trained specifically with boundary-sensitive loss functions (Dice Loss + Binary Cross-Entropy).
3. **Rigorous Clinical Validation**:
   - Test set benchmarked against histopathological biopsy results (CIN2+, CIN3+).
   - Independent external validation on varied imaging equipment and lighting settings.
