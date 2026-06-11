# Age Group Estimation from Faces — CNN Image Classification

A CNN-based image classifier that predicts a person's **age group** from a facial image, using the **UTKFace** dataset. The continuous age label is binned into 6 ordered age groups, making this a 6-class image-classification problem.

> Phase 2 deliverable — Machine Learning course (Applied CNN-based Image Classification).

## Age groups

| Class | Age range | Label |
|-------|-----------|-------|
| 0 | 0–12  | Child |
| 1 | 13–19 | Teenager |
| 2 | 20–34 | Young Adult |
| 3 | 35–49 | Adult |
| 4 | 50–64 | Middle-Aged |
| 5 | 65+   | Senior |

## Dataset

- **UTKFace** — 23,708 RGB face images (aligned and cropped version).
- Labels are stored in each filename as `[age]_[gender]_[race]_[datetime].jpg`; only the age field is used.
- Link: https://www.kaggle.com/datasets/jangedoo/utkface-new

The dataset is **not** included in this repository. Download it from the link above (or add it directly in a Kaggle notebook).

Class distribution (imbalanced — handled with balanced class weights):

| Age group | 0–12 | 13–19 | 20–34 | 35–49 | 50–64 | 65+ |
|-----------|------|-------|-------|-------|-------|-----|
| Images    | 3,413 | 1,180 | 9,634 | 4,492 | 3,031 | 1,958 |

## Repository structure

```
.
├── age_group_estimation.ipynb        # main notebook (full pipeline)
├── Proposal_AgeGroupEstimation.docx  # project proposal
├── README.md
└── figures/                          # output figures
    ├── class_distribution.png
    ├── sample_images.png
    ├── cnn_curves.png
    ├── cnn_confusion_matrix.png
    ├── mobilenet_curves.png
    ├── mobilenet_confusion_matrix.png
    ├── model_comparison.png
    └── gradcam.png
```

## What the notebook does

1. **Dataset loading** — robustly parses ages from filenames (skips malformed files).
2. **EDA** — class distribution and sample images per group.
3. **Preprocessing** — resize to 128×128, normalize to [0, 1], and augment the training set.
4. **Split** — stratified 70 / 15 / 15 train / validation / test.
5. **Class weighting** — to counter the dataset's imbalance.
6. **Custom CNN** — four Conv→BatchNorm→MaxPool blocks + dense head.
7. **Transfer learning** — MobileNetV2 (ImageNet), two-stage feature extraction + fine-tuning.
8. **Evaluation** — accuracy/loss curves, confusion matrix, classification report.
9. **Comparison** — custom CNN vs MobileNetV2.
10. **Grad-CAM** — visualizes the facial regions that drive predictions.

All figures are saved to `/kaggle/working` and are included in `figures/`.

## How to run

### On Kaggle (recommended)
1. Create a new notebook → **File → Import Notebook** → upload `age_group_estimation.ipynb`.
2. **Add Input → Datasets** → search "UTKFace" → add the UTKFace dataset (e.g. `jangedoo/utkface-new`).
3. **Settings → Accelerator → GPU T4 ×2**, and make sure **Internet is ON** (needed to download the MobileNetV2 ImageNet weights).
4. In the configuration cell, set `BASE` to the folder where the dataset mounted. To find it, run:
   ```python
   import glob, os
   print(os.path.dirname(glob.glob('/kaggle/input/**/UTKFace/*.jpg', recursive=True)[0]))
   ```
   then set `BASE` to its parent — for example:
   ```python
   BASE = '/kaggle/input/datasets/jangedoo/utkface-new'
   ```
5. **Run All**. The dataset-loading cell should print `Valid images parsed: 23708`. All figures are written to `/kaggle/working`.

### Locally
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow
```
Download and extract UTKFace, then set `BASE` in the configuration cell to the folder containing your local `UTKFace` directory and run all cells. A GPU is strongly recommended for training.

## Requirements

- Python 3.x
- TensorFlow / Keras, scikit-learn, pandas, numpy, matplotlib, seaborn, pillow

## Notes

- The label-to-index mapping is locked via `classes=CLASS_NAMES`, so the confusion matrix is always correctly aligned.
- Training uses `EarlyStopping` and `ReduceLROnPlateau`; the configured epoch counts are upper bounds.
- Internet must be enabled on Kaggle so the MobileNetV2 ImageNet weights can download.
