# Vehicle Classification Using YOLO11L

An image-level vehicle classification system based on **YOLO11L-cls**, trained using transfer learning on the **Vehicle Data Set** from Kaggle.

The project covers the complete machine-learning workflow from dataset ingestion and exploratory data analysis through preprocessing, model fine-tuning, test-set inference, quantitative evaluation, confusion analysis, and prediction visualisation.

---

## Table of Contents

- [Overview](#overview)
- [Objectives](#objectives)
- [Scope](#scope)
- [Dataset](#dataset)
  - [Dataset Source](#dataset-source)
  - [Original Dataset Structure](#original-dataset-structure)
  - [Vehicle Classes](#vehicle-classes)
  - [Class Imbalance](#class-imbalance)
  - [Dataset Filtering](#dataset-filtering)
- [Methodology](#methodology)
  - [Overall Pipeline](#overall-pipeline)
  - [Dataset Scanning](#dataset-scanning)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Preprocessing](#preprocessing)
  - [Stratified Splitting](#stratified-splitting)
  - [ImageFolder Preparation](#imagefolder-preparation)
- [Model](#model)
  - [YOLO11L-cls](#yolo11l-cls)
  - [Transfer Learning](#transfer-learning)
  - [Model Architecture](#model-architecture)
- [Training](#training)
  - [Training Configuration](#training-configuration)
  - [Data Augmentation](#data-augmentation)
  - [Training Procedure](#training-procedure)
- [Evaluation](#evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Test Inference](#test-inference)
  - [Confusion Matrices](#confusion-matrices)
  - [Per-Class Analysis](#per-class-analysis)
  - [Confidence Analysis](#confidence-analysis)
  - [Sample Predictions](#sample-predictions)
- [Results](#results)
  - [Experiment Summary](#experiment-summary)
  - [Performance](#performance)
  - [Training and Generalisation](#training-and-generalisation)
  - [Class-Level Observations](#class-level-observations)
- [Implementation](#implementation)
  - [Software Environment](#software-environment)
  - [Libraries](#libraries)
  - [Pipeline Components](#pipeline-components)
  - [System Working](#system-working)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Output Files](#output-files)
- [Reproducibility](#reproducibility)
- [Hardware and Runtime Environment](#hardware-and-runtime-environment)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Notes](#notes)
- [References](#references)

---

## Overview

This project implements a multi-class vehicle image classification pipeline using the **YOLO11L-cls** classification framework developed by Ultralytics.

The dataset contains vehicle images arranged by vehicle category. The original `train` and `test` directories are scanned and combined before preprocessing. Classes with fewer than 60 images are removed, after which the remaining images are divided into stratified training, validation, and test subsets.

The YOLO11L-cls model is initialized with pretrained weights and fine-tuned on the prepared dataset. The training configuration uses the AdamW optimizer, cosine learning-rate scheduling, warmup, partial backbone freezing, RandAugment, colour augmentation, random scaling, and random erasing.

The trained model is evaluated on a held-out test set using Top-1 and Top-5 accuracy, weighted and macro precision/recall/F1-score, confusion matrices, per-class metrics, confidence distributions, and sample prediction visualisations.

> **Task type:** image-level classification.  
> The project does not perform bounding-box localisation, multi-object detection, or object tracking.

---

## Objectives

The main objectives of the project are:

1. Scan and analyse a multi-class vehicle image dataset.
2. Perform exploratory data analysis on class distribution, image dimensions, aspect ratios, and representative samples.
3. Remove severely underrepresented classes using a minimum-sample threshold.
4. Construct reproducible stratified train, validation, and test partitions.
5. Organise the images into a classification-folder structure compatible with the YOLO classification trainer.
6. Fine-tune a pretrained YOLO11L-cls model using transfer learning.
7. Use a controlled training configuration with AdamW, cosine learning-rate scheduling, warmup, frozen backbone layers, and data augmentation.
8. Evaluate the trained model using overall and per-class classification metrics.
9. Analyse misclassification patterns using raw and normalised confusion matrices.
10. Analyse the relationship between class support and F1-score.
11. Examine prediction confidence for correct and incorrect classifications.
12. Produce representative test-set prediction visualisations.

---

## Scope

The implementation is designed for **static image-level vehicle classification**.

### Included

- Dataset ingestion
- Dataset scanning
- Exploratory data analysis
- Class filtering
- Stratified data splitting
- ImageFolder dataset construction
- Transfer learning
- YOLO11L-cls training
- Validation
- Test-set inference
- Classification metrics
- Confusion-matrix analysis
- Per-class performance analysis
- Confidence analysis
- Prediction visualisation

### Not included

- Real-time video processing
- Multi-object detection within a scene
- Bounding-box prediction
- Multi-object tracking
- Edge-device deployment
- Production inference serving
- Model export pipeline
- Ablation experiments

---

# Dataset

## Dataset Source

The project uses the **Vehicle Data Set** published on Kaggle by **Sandeep Prasad (`iamsandeepprasad`)**.

Dataset page:

```text
https://www.kaggle.com/datasets/iamsandeepprasad/vehicle-data-set
```

The dataset is publicly available and is organised as an image classification dataset with class-specific folders.

The dataset itself is **not included in this repository**.

---

## Original Dataset Structure

The source dataset contains separate `train` and `test` directories.

A simplified representation is:

```text
vehicle-data-set/
├── train/
│   ├── Ambulance/
│   ├── Barge/
│   ├── Bicycle/
│   ├── Boat/
│   ├── Bus/
│   ├── Car/
│   ├── Cart/
│   ├── Caterpillar/
│   ├── Helicopter/
│   ├── Limousine/
│   ├── Motorcycle/
│   ├── Segway/
│   ├── Snowmobile/
│   ├── Tank/
│   ├── Taxi/
│   ├── Truck/
│   └── Van/
└── test/
    ├── Ambulance/
    ├── Barge/
    ├── Bicycle/
    ├── Boat/
    ├── Bus/
    ├── Car/
    ├── Cart/
    ├── Caterpillar/
    ├── Helicopter/
    ├── Limousine/
    ├── Motorcycle/
    ├── Segway/
    ├── Snowmobile/
    ├── Tank/
    ├── Taxi/
    ├── Truck/
    └── Van/
```

The original dataset contains approximately **28,045 images across 17 classes**.

The original train/test organisation is not used directly for the final experiment. Both portions are scanned and pooled before the custom stratified split is created.

---

## Vehicle Classes

The original dataset contains the following 17 vehicle categories:

- Ambulance
- Barge
- Bicycle
- Boat
- Bus
- Car
- Cart
- Caterpillar
- Helicopter
- Limousine
- Motorcycle
- Segway
- Snowmobile
- Tank
- Taxi
- Truck
- Van

Each image receives its class label from the name of its parent directory.

No bounding-box annotations are used because the dataset is intended for image-level classification.

---

## Class Imbalance

The raw dataset is substantially imbalanced.

Some classes contain thousands of images while some categories contain only a small number of samples. The report identifies **Boat** as one of the largest classes, with more than 8,000 images, while **Cart** contains only **51 images**.

This imbalance affects both model training and evaluation. A model can obtain strong overall performance while still performing poorly on classes with limited support.

For this reason, the preprocessing pipeline applies a minimum class-size threshold before constructing the final dataset.

---

## Dataset Filtering

A minimum sample threshold of:

```python
MIN_SAMPLES = 60
```

is applied.

Classes containing fewer than 60 images are excluded from the experiment.

In the documented experiment:

```text
Cart: 51 images
```

was removed.

After filtering:

```text
Original classes: 17
Retained classes: 16
```

The filtered dataset contains approximately:

```text
27,994 images
```

across the 16 retained vehicle classes.

---

# Methodology

## Overall Pipeline

The complete workflow is:

```text
Kaggle Vehicle Data Set
        │
        ▼
Dataset Scanning
        │
        ▼
Class Distribution Analysis
        │
        ├── Image Dimension Analysis
        ├── Aspect Ratio Analysis
        └── Sample Image Analysis
        │
        ▼
Filter Classes < 60 Images
        │
        ▼
Stratified 75 / 15 / 10 Split
        │
        ▼
ImageFolder Dataset Construction
        │
        ▼
YOLO11L-cls + Pretrained Weights
        │
        ▼
Fine-tuning on GPU
        │
        ▼
Training Curves
        │
        ▼
Best Checkpoint
        │
        ▼
Test-set Inference
        │
        ├── Top-1 / Top-5
        ├── Precision / Recall / F1
        ├── Confusion Matrices
        ├── Per-Class Metrics
        ├── F1 vs Support
        ├── Confidence Analysis
        └── Sample Predictions
```

---

## Dataset Scanning

A custom `scan_dataset()` function recursively traverses the original `train/` and `test/` directories.

For each supported image:

- `.jpg`
- `.jpeg`
- `.png`

the function records:

```text
path
class
```

The class is obtained from the image's parent directory.

The resulting records are stored in a pandas DataFrame.

Conceptually:

```python
def scan_dataset(base_dir):
    records = []

    for split in ["train", "test"]:
        split_dir = os.path.join(base_dir, split)

        for root, _, files in os.walk(split_dir):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, file)
                    cls = os.path.basename(root)

                    records.append({
                        "path": path,
                        "class": cls
                    })

    return pd.DataFrame(records)
```

The complete implementation additionally handles the required dataset paths and subsequent processing.

---

## Exploratory Data Analysis

EDA is performed before model training.

The analysis includes:

### 1. Raw Class Distribution

The number of images per class is calculated using the class labels obtained during dataset scanning.

The distribution is visualised as a horizontal bar chart.

Output:

```text
eda_raw_distribution.png
```

This analysis exposes the substantial class imbalance present in the raw dataset.

### 2. Image Dimensions and Aspect Ratios

A random sample of 300 images is inspected using Pillow.

The width and height of the sampled images are collected and visualised.

The report records approximate mean dimensions of:

```text
Mean width  ≈ 1,117 pixels
Mean height ≈   769 pixels
```

The native image dimensions vary considerably because the images originate from heterogeneous online sources.

Output:

```text
eda_dimensions.png
```

### 3. Representative Class Images

One representative image per class is displayed to inspect visual diversity.

Output:

```text
eda_sample_images.png
```

### 4. Split Distribution

After the stratified split, the number of samples belonging to each class in each partition is visualised.

Output:

```text
eda_split_distribution.png
```

---

## Preprocessing

The preprocessing stage consists of:

1. Scan all available images.
2. Build the dataset DataFrame.
3. Count samples per class.
4. Remove classes below `MIN_SAMPLES = 60`.
5. Pool the original train and test records.
6. Perform a stratified split.
7. Create the classification directory structure.
8. Copy images into their corresponding split/class folders.

The resulting directory is compatible with the Ultralytics classification trainer.

---

## Stratified Splitting

The filtered dataset is split into:

| Partition | Percentage |
|---|---:|
| Training | 75% |
| Validation | 15% |
| Test | 10% |

The split is performed using `train_test_split` from scikit-learn.

The class label is supplied through the `stratify` argument so that class proportions are preserved across the partitions.

The random seed is fixed at:

```python
SEED = 42
```

The original train/test split supplied with the dataset is therefore replaced by the project's own three-way stratified split.

---

## ImageFolder Preparation

The YOLO classification trainer reads class membership from the directory structure.

The prepared dataset follows the form:

```text
classification_dataset/
├── train/
│   ├── Ambulance/
│   ├── Barge/
│   ├── ...
│   └── Van/
├── val/
│   ├── Ambulance/
│   ├── Barge/
│   ├── ...
│   └── Van/
└── test/
    ├── Ambulance/
    ├── Barge/
    ├── ...
    └── Van/
```

No YAML configuration file is required for this classification setup.

Images are copied using `shutil.copy`, with `tqdm` used for progress tracking.

Filename collisions are handled by appending a random integer suffix when necessary.

---

# Model

## YOLO11L-cls

The project uses the large classification variant of YOLO11:

```text
YOLO11L-cls
```

The pretrained checkpoint is:

```text
yolo11l-cls.pt
```

The model is loaded using the Ultralytics interface:

```python
from ultralytics import YOLO

model = YOLO("yolo11l-cls.pt")
```

The model is used as an **image classification network**, not as a conventional YOLO object detector.

Each input image produces class probabilities, from which the top-ranked vehicle category is selected.

---

## Transfer Learning

The model is initialized from pretrained YOLO11L-cls weights rather than trained from random initialization.

The first 10 backbone layers are frozen:

```text
freeze = 10
```

This keeps lower-level feature representations fixed while allowing the upper portion of the network and classification head to adapt to the vehicle dataset.

The approach is intended to preserve useful pretrained representations such as edges, textures, and shapes while adapting the model to the 16 vehicle categories used in the experiment.

---

## Model Architecture

The YOLO11L-cls network contains a feature-extraction backbone followed by a classification head.

The classification head maps the learned feature representation to one logit for each target vehicle class.

The highest-scoring class is used as the Top-1 prediction.

The documented trained model contains:

```text
14,115,624 parameters
```

The architecture summary is obtained using:

```python
model.info(verbose=False)
```

Parameter counts are calculated from the model parameter shapes.

---

# Training

## Training Configuration

The documented experiment uses the following configuration:

| Parameter | Value |
|---|---:|
| Model | YOLO11L-cls |
| Image size | 384 × 384 |
| Epochs | 25 |
| Batch size | 24 |
| Optimizer | AdamW |
| Initial learning rate (`lr0`) | 1e-4 |
| Final LR factor (`lrf`) | 0.1 |
| Cosine LR scheduling | Enabled |
| Warmup epochs | 3.0 |
| Weight decay | 1e-4 |
| Dropout | 0.1 |
| Frozen layers | 10 |
| Early stopping patience | 15 |
| Random seed | 42 |
| Cache | RAM |
| Device | GPU / device 0 |

---

## Data Augmentation

Training uses several augmentation strategies:

| Augmentation | Configuration |
|---|---:|
| RandAugment | Enabled |
| Hue | 0.015 |
| Saturation | 0.4 |
| Value | 0.3 |
| Horizontal flip | 0.5 |
| Vertical flip | 0.0 |
| Scale | 0.5 |
| Random erasing | 0.25 |

The corresponding Ultralytics configuration includes:

```python
auto_augment="randaugment"
hsv_h=0.015
hsv_s=0.4
hsv_v=0.3
fliplr=0.5
scale=0.5
erasing=0.25
```

These transformations increase the effective visual diversity of the training data and are intended to improve robustness to changes in colour, viewpoint, scale, and partial occlusion.

---

## Optimisation Strategy

### AdamW

The optimizer is:

```text
AdamW
```

with:

```text
learning rate = 1e-4
weight decay  = 1e-4
```

AdamW provides adaptive gradient updates while applying decoupled weight decay.

### Cosine Learning Rate Schedule

Cosine scheduling is enabled:

```python
cos_lr=True
```

with:

```text
lr0 = 1e-4
lrf = 0.1
```

The learning rate gradually decreases during training.

### Warmup

A warmup period of:

```text
3 epochs
```

is used to avoid aggressive parameter updates during the early stage of fine-tuning.

### Early Stopping

Training uses:

```text
patience = 15
```

so that training can stop when validation performance stops improving.

### RAM Caching

The experiment uses:

```python
cache="ram"
```

to reduce repeated disk I/O during training.

---

## Training Procedure

The core training call follows the documented configuration:

```python
model.train(
    data=CLS_DIR,
    epochs=25,
    imgsz=384,
    batch=24,
    patience=15,
    freeze=10,
    optimizer="AdamW",
    lr0=1e-4,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=3.0,
    weight_decay=1e-4,
    dropout=0.1,
    auto_augment="randaugment",
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    scale=0.5,
    erasing=0.25,
    cache="ram",
    seed=42
)
```

Training artefacts are stored under:

```text
/kaggle/working/runs/cls_yolo11L/
```

Important outputs include:

```text
weights/best.pt
weights/last.pt
results.csv
```

---

# Evaluation

## Evaluation Metrics

The evaluation stage uses several complementary metrics.

### Top-1 Accuracy

Measures the proportion of test images for which the highest-confidence predicted class matches the true class.

### Top-5 Accuracy

Measures the proportion of test images for which the correct class appears within the model's five highest-ranked predictions.

### Weighted Precision, Recall and F1

Per-class metrics are weighted according to the number of true samples in each class.

These metrics reflect performance over the observed test distribution while accounting for class support.

### Macro Precision, Recall and F1

Per-class metrics are averaged equally.

Macro metrics are therefore more sensitive to performance on minority classes.

### Per-Class Metrics

The complete scikit-learn classification report provides:

```text
precision
recall
F1-score
support
```

for every retained vehicle class.

### Confidence Distribution

Prediction confidence is separated into:

```text
Correct predictions
Incorrect predictions
```

The mean confidence of both groups is reported.

### Confusion Matrix

Two confusion matrices are generated:

1. Raw-count confusion matrix
2. Row-normalised confusion matrix

The normalised matrix represents class-wise recall.

---

## Test Inference

The best checkpoint is loaded:

```python
best = YOLO("/kaggle/working/runs/cls_yolo11L/weights/best.pt")
```

The test images are processed in batches of:

```text
24 images
```

For every test image, the pipeline records:

- True class
- Top-1 predicted class
- Top-1 confidence
- Top-5 predicted classes

These values are accumulated for the final evaluation.

---

## TensorFlow Cross-Check

A TensorFlow `tf.data` pipeline is also constructed for the test set.

The image-loading process uses operations including:

```python
tf.io.read_file
tf.image.decode_jpeg
tf.image.resize
```

The resulting labels and predictions are additionally checked using:

```python
tf.keras.metrics.SparseCategoricalAccuracy
```

This provides an independent cross-check of the primary classification accuracy calculation.

---

## Confusion Matrices

The evaluation pipeline produces:

```text
results_confusion_matrix.png
results_confusion_matrix_normalized.png
```

### Raw Confusion Matrix

Shows the absolute number of predictions assigned to each predicted class for every true class.

### Normalised Confusion Matrix

Each row is normalised by the number of samples belonging to the corresponding true class.

This makes class-wise recall easier to compare despite different class supports.

---

## Per-Class Analysis

The project generates a grouped bar chart containing:

```text
Precision
Recall
F1-score
```

for every retained vehicle class.

Output:

```text
results_per_class_metrics.png
```

The complete classification report is also saved as:

```text
results_classification_report.csv
```

---

## F1 vs Support

The project analyses the relationship between class support and model performance.

For each class:

```text
x-axis = test-set support
y-axis = F1-score
```

The class name is annotated on the plot.

Output:

```text
results_f1_vs_support.png
```

This analysis is useful for identifying whether low sample counts are associated with weaker class-level performance.

---

## Confidence Analysis

The confidence analysis separates predictions into correct and incorrect groups.

Output:

```text
results_confidence_distribution.png
```

The documented experiment reports:

```text
Mean confidence — correct predictions:   91.83%
Mean confidence — incorrect predictions: 66.69%
Difference:                               25.14 percentage points
```

---

## Sample Predictions

The evaluation pipeline creates a 2 × 3 grid containing:

- Three correctly classified examples
- Three incorrectly classified examples

Each image includes:

```text
True label
Predicted label
Confidence score
```

Output:

```text
results_sample_predictions.png
```

---

## Top Confused Class Pairs

The confusion matrix is also used to identify the eight largest off-diagonal error counts.

This produces a list of the class pairs that are most frequently confused by the model.

This analysis helps identify visually similar vehicle categories that require additional data or model refinement.

---

# Results

## Experiment Summary

The documented experiment was trained for:

```text
25 epochs
```

using a Kaggle GPU environment.

The final configuration used:

```text
Model:             YOLO11L-cls
Parameters:        14,115,624
Input resolution:  384 × 384
Retained classes:  16
Images:            approximately 27,994
Train split:       75%
Validation split:  15%
Test split:        10%
Frozen layers:     10
```

The original `Cart` class was removed because it contained only 51 images.

---

## Performance

The documented test-set results are:

| Metric | Result |
|---|---:|
| Top-1 Accuracy | **89.00%** |
| Top-5 Accuracy | **99.32%** |
| Weighted F1-score | **88.69%** |
| Macro F1-score | **79.03%** |
| Mean confidence — correct | **91.83%** |
| Mean confidence — incorrect | **66.69%** |

The report also notes a **10.32 percentage-point difference** between Top-1 and Top-5 accuracy.

The weighted F1-score is close to the Top-1 accuracy, while the lower macro F1-score reflects weaker performance on some minority classes.

---

## Training and Generalisation

Training and validation cross-entropy loss decrease over the 25-epoch training period.

The final train-validation loss gap reported by the experiment is:

```text
0.062
```

The training and validation curves show broadly similar behaviour throughout training.

The training configuration combines:

- Partial backbone freezing
- AdamW weight decay
- RandAugment
- HSV augmentation
- Random scaling
- Random erasing
- Early stopping

These components were used to control overfitting during fine-tuning.

---

## Class-Level Observations

The documented classification report identifies several differences in class-level performance.

Examples of higher F1-scores include:

| Class | F1-score |
|---|---:|
| Helicopter | 0.9538 |
| Motorcycle | 0.9416 |
| Boat | 0.9404 |
| Bus | 0.9275 |
| Snowmobile | 0.9231 |

Examples of weaker performance include:

| Class | F1-score |
|---|---:|
| Barge | 0.1481 |
| Limousine | 0.5714 |
| Ambulance | 0.5926 |

The report identifies limited class support as an important factor for the weakest categories.

For example:

```text
Barge       → 20 test samples
Limousine   →  7 test samples
Ambulance   → 13 test samples
```

Barge has a reported recall of only `0.10`.

The F1-versus-support analysis also indicates substantial variation among classes with small test support.

---

# Implementation

## Software Environment

The project is implemented in:

```text
Python 3
```

and was executed in a:

```text
Kaggle Jupyter Notebook
```

The training stage uses a Kaggle-provided NVIDIA GPU.

---

## Libraries

The implementation uses the following libraries and frameworks:

| Library | Purpose |
|---|---|
| Ultralytics YOLO11 | Model, training and prediction |
| TensorFlow | GPU detection, reproducibility, image pipeline and metric cross-check |
| Pillow | Image dimension analysis |
| NumPy | Numerical operations and random seed control |
| pandas | Dataset DataFrame and tabular analysis |
| Matplotlib | Visualisation |
| Seaborn | Confusion-matrix visualisation |
| scikit-learn | Stratified splitting and classification metrics |
| tqdm | Progress bars |

---

## Pipeline Components

### `scan_dataset()`

Scans the source dataset and creates a DataFrame containing:

```text
path
class
```

### Class Filtering

Counts images per class and removes categories with:

```text
count < 60
```

### Stratified Split

Creates:

```text
75% train
15% validation
10% test
```

while preserving class proportions.

### EDA

Generates:

```text
eda_raw_distribution.png
eda_dimensions.png
eda_sample_images.png
eda_split_distribution.png
```

### ImageFolder Construction

Copies images into:

```text
train/class/
val/class/
test/class/
```

### Model Summary

Loads:

```text
yolo11l-cls.pt
```

and prints the architecture summary and parameter count.

### Training

Runs the YOLO11L-cls training configuration described above.

### Training Curves

Reads:

```text
results.csv
```

and produces a three-panel training figure containing:

- Training/validation loss
- Top-1/Top-5 accuracy
- Learning-rate schedule

Output:

```text
results_training_curves.png
```

### Validation

Loads:

```text
best.pt
```

and evaluates the validation split.

### Test Inference

Runs the best checkpoint over the test images and records class predictions and confidence scores.

### Metrics

Computes:

```text
Top-1
Top-5
Weighted precision
Weighted recall
Weighted F1
Macro precision
Macro recall
Macro F1
```

and the confidence statistics.

### Confusion Analysis

Generates both raw and normalised confusion matrices.

### Per-Class Analysis

Generates the classification report and per-class metric chart.

### F1-Support Analysis

Plots class F1 against test support.

### Confidence Analysis

Compares confidence distributions for correct and incorrect predictions.

### Prediction Visualisation

Produces the 2 × 3 sample prediction grid.

### Run Summary

Prints the final model information, dataset counts, training outcome, metrics, and generated output files.

---

## System Working

The implementation operates sequentially in the notebook.

### Stage 1 — Data Ingestion

The source directory is scanned and a unified DataFrame is created from the original train and test folders.

### Stage 2 — EDA

The dataset is analysed to determine:

- Number of images
- Number of classes
- Class imbalance
- Native image dimensions
- Aspect ratios
- Representative image appearance

### Stage 3 — Filtering

Classes below the minimum sample threshold are removed.

### Stage 4 — Splitting

The filtered dataset is divided into stratified train, validation, and test partitions.

### Stage 5 — Dataset Preparation

Images are copied into the ImageFolder directory layout.

### Stage 6 — Model Initialisation

The pretrained YOLO11L-cls checkpoint is loaded.

### Stage 7 — Fine-Tuning

The model is trained using the specified optimisation and augmentation configuration.

### Stage 8 — Validation

The best checkpoint is evaluated on the validation set.

### Stage 9 — Test Inference

The best checkpoint predicts every test image.

### Stage 10 — Evaluation

Predictions are compared with ground-truth labels.

### Stage 11 — Visualisation

The evaluation pipeline produces confusion matrices, metric charts, confidence plots, and sample predictions.

---

# Repository Structure

A typical repository layout is:

```text
vehicle-detection-yolo11l/
│
├── data/
│   ├── raw/
│   │   └── vehicle-data-set/
│   └── processed/
│
├── docs/
│   ├── vehicle_detection_report.docx
│   └── CODE_ALIGNMENT.md
│
├── notebooks/
│   └── vehicle_detection_report_aligned.ipynb
│
├── reports/
│   ├── eda/
│   │   ├── eda_raw_distribution.png
│   │   ├── eda_dimensions.png
│   │   ├── eda_sample_images.png
│   │   └── eda_split_distribution.png
│   │
│   └── evaluation/
│       ├── results_training_curves.png
│       ├── results_confusion_matrix.png
│       ├── results_confusion_matrix_normalized.png
│       ├── results_per_class_metrics.png
│       ├── results_classification_report.csv
│       ├── results_f1_vs_support.png
│       ├── results_confidence_distribution.png
│       └── results_sample_predictions.png
│
├── src/
│   ├── report_pipeline.py
│   └── report_pipeline_local.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

The exact contents may vary depending on whether generated datasets, model weights, and experiment outputs are retained locally.

---

# Installation

## 1. Clone the Repository

```bash
git clone <repository-url>
cd vehicle-detection-yolo11l
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install a compatible GPU-enabled deep-learning environment appropriate for the target machine.

---

# Running the Project

## Kaggle Notebook

The documented experiment was executed in Kaggle.

The notebook should be run sequentially because later stages depend on the datasets and outputs produced by earlier stages.

The primary workflow is:

```text
Dataset
→ EDA
→ Filtering
→ Stratified split
→ ImageFolder construction
→ YOLO11L-cls training
→ Validation
→ Test inference
→ Evaluation
```

---

## Local Execution

The repository can also be used as a local project, provided the dataset is available and the required Python dependencies are installed.

Place the dataset under:

```text
data/raw/vehicle-data-set/
```

with:

```text
data/raw/vehicle-data-set/
├── train/
└── test/
```

The training process requires substantially more resources than a lightweight CPU-only Python workflow because YOLO11L is a large classification model.

---

# Output Files

## EDA Outputs

```text
eda_raw_distribution.png
eda_dimensions.png
eda_sample_images.png
eda_split_distribution.png
```

### `eda_raw_distribution.png`

Raw class-frequency distribution.

### `eda_dimensions.png`

Sampled image width, height, and aspect-ratio analysis.

### `eda_sample_images.png`

Representative image from each class.

### `eda_split_distribution.png`

Class distribution across train, validation, and test partitions.

---

## Training Outputs

The documented Kaggle run stores training artefacts under:

```text
/kaggle/working/runs/cls_yolo11L/
```

Important files include:

```text
weights/best.pt
weights/last.pt
results.csv
```

### `best.pt`

Best model checkpoint selected by the training process.

### `last.pt`

Checkpoint from the final training epoch.

### `results.csv`

Epoch-by-epoch training and validation metrics.

---

## Evaluation Outputs

```text
results_training_curves.png
results_confusion_matrix.png
results_confusion_matrix_normalized.png
results_per_class_metrics.png
results_classification_report.csv
results_f1_vs_support.png
results_confidence_distribution.png
results_sample_predictions.png
```

---

# Reproducibility

The documented experiment uses:

```text
SEED = 42
```

The seed is applied across the relevant Python, NumPy, TensorFlow, and Ultralytics components.

Important experiment settings are:

```text
Minimum samples per class: 60

Train / validation / test:
75% / 15% / 10%

Model:
YOLO11L-cls

Image size:
384 × 384

Epochs:
25

Batch size:
24

Optimizer:
AdamW

Initial learning rate:
1e-4

Frozen layers:
10
```

Exact results can vary if any of the following change:

- Dataset version
- Dataset contents
- Ultralytics version
- Python version
- TensorFlow version
- GPU hardware
- CUDA environment
- Other dependency versions
- Training configuration

The values reported in this README refer specifically to the experiment documented in the project report.

---

# Hardware and Runtime Environment

The documented training run was performed in a:

```text
Kaggle GPU environment
```

using:

```text
device = 0
```

The dataset is cached in RAM during training:

```python
cache="ram"
```

This reduces repeated disk access during training.

The project is designed around GPU-accelerated training rather than CPU-only training.

---

# Limitations

## Image-Level Classification

The system predicts a vehicle category for an image containing a dominant vehicle subject.

It does not provide:

- Bounding boxes
- Object localisation
- Multiple object instances
- Object tracking

A scene containing several vehicles would require an object-detection pipeline with instance-level annotations.

---

## Static Image Evaluation

The current pipeline evaluates individual images independently.

It does not use temporal information from video frames.

A video-based system would require additional components for:

- Frame processing
- Multi-object tracking
- Temporal consistency
- Real-time inference

---

## Rare-Class Filtering

Classes with fewer than 60 images are removed.

This improves the reliability of the remaining class-level evaluation but also reduces the number of vehicle categories represented by the final model.

The documented experiment therefore uses 16 classes rather than the original 17.

---

## No Deployment Pipeline

The project does not include:

- REST API serving
- Web application deployment
- Model serving infrastructure
- ONNX/TensorRT export
- Edge-device deployment

The documented model remains within the training/evaluation workflow.

---

## No Augmentation Ablation

The experiment uses several augmentation methods simultaneously.

It does not isolate the individual contribution of:

- RandAugment
- HSV augmentation
- Random scaling
- Random erasing

through formal ablation experiments.

---

## Dataset Imbalance

Although filtering removes the most severely underrepresented class, the retained classes are still not uniformly distributed.

Consequently, macro-level metrics remain lower than weighted metrics in the documented experiment.

---

# Future Improvements

## Bounding-Box Detection

A natural extension is to replace image-level classification with object detection using precise bounding-box annotations.

Suitable annotation tools include:

- Roboflow
- CVAT

This would allow the system to localise multiple vehicles within a scene.

---

## Larger Dataset

Additional images can be collected for underrepresented vehicle categories.

Particular attention can be given to classes such as:

- Limousine
- Snowmobile
- Ambulance
- Tank
- Barge

Increasing the support of these classes could provide more reliable training and evaluation.

---

## Synthetic Data

Synthetic images generated through 3D rendering or generative methods could be used to increase the training diversity of rare classes.

---

## Ablation Studies

Future experiments can compare:

- Different augmentation configurations
- Different freeze depths
- Different learning rates
- Different model sizes
- Training with and without RandAugment
- Training with and without random erasing

This would provide a more controlled analysis of the contribution of each training component.

---

## Model Export and Deployment

The trained model can be extended toward practical deployment through:

- Model export
- GPU inference serving
- REST API integration
- Edge-device optimisation
- Real-time camera inference

---

## Video and Tracking

A future system can combine vehicle detection with tracking to support:

- Vehicle counting
- Traffic-flow analysis
- Multi-frame tracking
- Vehicle-type statistics
- Real-time monitoring

---

# Notes

- The dataset is not included in the repository.
- Model weights are not included by default because of their size.
- Generated training and evaluation files can be excluded from Git using `.gitignore`.
- Dataset licensing and usage conditions should be checked before redistribution or commercial use.
- The project is an image-classification system even though the broader project terminology may refer to it as vehicle detection.
- The reported results correspond to the specific experiment documented in the project report.
- Re-running the pipeline with a different dataset version or software environment may produce different results.

---

# References

## Dataset

Sandeep Prasad, **Vehicle Data Set**, Kaggle.

```text
https://www.kaggle.com/datasets/iamsandeepprasad/vehicle-data-set
```

## YOLO11

Ultralytics, **YOLO11** documentation and model framework.

```text
https://docs.ultralytics.com/
```

## Python Libraries

The implementation uses:

- Ultralytics
- TensorFlow
- Pillow
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- tqdm

---

# Project Summary

The project implements a complete vehicle image-classification workflow using YOLO11L-cls.

The final documented experiment uses a filtered dataset of approximately **27,994 images across 16 vehicle classes**, a stratified **75/15/10 train-validation-test split**, and a pretrained YOLO11L-cls model fine-tuned for **25 epochs** at **384 × 384** resolution.

The experiment reports:

```text
Top-1 Accuracy       : 89.00%
Top-5 Accuracy       : 99.32%
Weighted F1-score    : 88.69%
Macro F1-score       : 79.03%
Correct confidence   : 91.83%
Incorrect confidence : 66.69%
```

The evaluation pipeline extends beyond overall accuracy by analysing class-level performance, confusion patterns, support-dependent behaviour, and prediction confidence.

