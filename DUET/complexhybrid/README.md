# Complex Hybrid Pipeline

This folder contains the detector ensemble workflow for RoadVision DUET.

Pipeline:

1. Train a YOLO11-L teacher.
2. Train an RT-DETR-L teacher.
3. Generate pseudo labels from one or both teachers.
4. Retrain detectors on the pseudo-labeled dataset.
5. Fuse YOLO11-L and RT-DETR-L predictions with WBF.
6. Write the final Kaggle submission CSV.

## Default Dataset

The best RoadVision split in this repo is:

- `DUET/RoadVision_DUET/Modified_YOLO_Effective/data_train_all.yaml`

It uses all 810 labeled images for training and keeps a controlled validation split for tuning.

## Scripts

- `train_yolo11l.py`
- `train_rtdetrl.py`
- `build_pseudo_dataset.py`
- `retrain_detector.py`
- `generate_submission.py`

## Output Folder Suggestion

Use these paths for clean organization:

- `DUET/complexhybrid/runs/yolo11l_modified_all`
- `DUET/complexhybrid/runs/rtdetrl_modified_all`
- `DUET/complexhybrid/pseudo_dataset`
- `DUET/complexhybrid/submission.csv`
