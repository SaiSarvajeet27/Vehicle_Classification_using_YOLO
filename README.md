# Vehicle Detection / Classification Using YOLO11L-cls

This repository contains the report-aligned implementation of the vehicle image-classification project.

> **Important:** The report calls the project "vehicle detection", but the implemented model is **YOLO11L-cls**, an image-classification model. It predicts a vehicle class for each image and does not produce bounding boxes for multiple vehicles.

## Canonical report-matching code

The code that should be compared directly with the screenshots in the report is:

```text
src/report_pipeline.py
```

It deliberately preserves the notebook-style structure and variable names shown in the screenshots.

For running with a normal Python interpreter rather than a Kaggle notebook, use:

```text
src/report_pipeline_local.py
```

The only intentional source-level difference between those two files is removal of the notebook-only `!pip install ultralytics tensorflow -q` line.

## Report-aligned sequence

The implementation follows the screenshots in this order:

1. Install/import dependencies and set random seeds.
2. Define `BASE_DIR`, `CLS_DIR`, and training constants.
3. Print environment/GPU information.
4. `scan_dataset()` over the original `train/` and `test/` folders.
5. Raw class-distribution EDA.
6. Image dimension/aspect-ratio EDA.
7. One sample image per class.
8. Remove classes with fewer than 60 images.
9. Stratified 75% / 15% / 10% train/validation/test split.
10. Copy images into the YOLO classification ImageFolder layout.
11. Build `YOLO('yolo11l-cls.pt')` and print model information.
12. Train with the hyperparameters shown in the report.
13. Read `results.csv` and plot loss, Top-1/Top-5 accuracy, and learning rate.
14. Load `best.pt` and validate.
15. Run batched test inference.
16. Calculate Top-1, Top-5, weighted/macro precision-recall-F1, confidence statistics, and the TensorFlow cross-check.
17. Generate confusion matrices, classification report, per-class metrics, F1-vs-support, confidence distribution, sample predictions, and confused class pairs.
18. Print the final run summary and saved files.

## Running in Kaggle

The report uses these paths:

```python
BASE_DIR = Path("/kaggle/input/datasets/iamsandeepprasad/vehicle-data-set/cardataset")
CLS_DIR = Path("/kaggle/working/vehicle_cls_dataset")
```

So the simplest reproduction is to put the dataset in Kaggle and run `src/report_pipeline.py` in notebook cells. The first line is notebook-only installation magic.

## Running locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Then edit `BASE_DIR` in `src/report_pipeline_local.py` to point to your extracted dataset and run it. The rest of the code intentionally retains the report's `/kaggle/working/...` output paths, so those should also be changed if you are not using Kaggle.

## Dataset

The report references the Kaggle Vehicle Data Set by `iamsandeepprasad`. The dataset itself is not included in this repository.

## Results

The numerical results in the report (for example, 89.00% Top-1 and 99.32% Top-5) belong to the original reported run. They are not hard-coded into the implementation and should not be claimed as reproduced until the code is run against the same data/environment.

## Documentation

- `docs/vehicle_detection_report.docx` — project report.
- `docs/CODE_ALIGNMENT.md` — exact mapping of the repository code to the screenshots.
