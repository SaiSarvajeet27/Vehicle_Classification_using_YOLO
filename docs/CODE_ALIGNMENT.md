# Code alignment with the report screenshots

The canonical implementation is `src/report_pipeline.py`.

It intentionally follows the report screenshots rather than the earlier modularized implementation:

1. Same import stack and seed setup.
2. Same `BASE_DIR`, `CLS_DIR`, `MIN_SAMPLES`, `IMG_SIZE`, `EPOCHS`, `BATCH`, `PATIENCE`, `VAL_FRAC`, and `TEST_FRAC` variables.
3. Same `scan_dataset()` structure.
4. Same EDA sequence and output filenames.
5. Same `< 60` class filtering and stratified 75/15/10 split.
6. Same `CLS_DIR` ImageFolder copying logic and collision handling.
7. Same YOLO11L-cls model construction and `model_yolo.info()` block.
8. Same training hyperparameters and `/kaggle/working/runs/cls_yolo11L` output location.
9. Same training-results plotting and train/validation loss-gap calculation.
10. Same `best.pt` validation call and TensorFlow image preprocessing function.
11. Same batched YOLO test inference, top-1/top-5 metrics, sklearn metrics, and TensorFlow top-1 cross-check.
12. Same confusion matrix, classification report, per-class chart, F1/support plot, confidence distribution, sample predictions, confused-pair output, and run summary.

`report_pipeline_local.py` is the same code with the Kaggle `!pip install ...` notebook magic removed so that it can be parsed by a normal Python interpreter. The default paths remain the report's Kaggle paths; change `BASE_DIR` and output locations when running outside Kaggle.

The report's screenshots are the source of truth for this alignment. No attempt is made here to silently redesign the code into a different architecture.
