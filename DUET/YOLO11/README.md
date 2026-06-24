# RoadVision DUET YOLO Baseline

This folder keeps the first Ultralytics YOLO pipeline isolated under `DUET/YOLO11`.

## 1. Prepare YOLO data

```powershell
python DUET/YOLO11/prepare_yolo_dataset.py
```

Output:

- `DUET/YOLO11/yolo_dataset/images/train`
- `DUET/YOLO11/yolo_dataset/images/val`
- `DUET/YOLO11/yolo_dataset/images/test`
- `DUET/YOLO11/yolo_dataset/labels/train`
- `DUET/YOLO11/yolo_dataset/labels/val`
- `DUET/YOLO11/yolo_dataset/data.yaml`
- `DUET/YOLO11/yolo_dataset/dataset_stats.json`

## 2. Train a first model

```powershell
python DUET/YOLO11/train_yolo.py --model yolo11m.pt --epochs 120 --imgsz 960 --batch 8 --device 0
```

For YOLO26x, use the local weight file or URL:

```powershell
python DUET/YOLO11/train_yolo.py --model https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt --epochs 140 --imgsz 1024 --batch 4 --name roadvision_yolo26x
```

## 3. Create a submission

```powershell
python DUET/YOLO11/predict_submission.py --weights DUET/YOLO11/runs/roadvision_yolo11m/weights/best.pt --output DUET/YOLO11/submission.csv
```

## Notes

- The submission `image_id` values should be the bare image stem without an extension.
- Submission confidence must be `0-1`. The included `sample_submission.csv` appears to contain some percentage-style confidence values, so this script writes normalized YOLO confidences.
- A 95% mAP target is ambitious with 810 labeled images and heavy domain shift. Start with this baseline, then improve via larger weights, higher image size, fold ensembling, pseudo-labeling test/unlabeled data, and class-aware augmentation for rare classes.
