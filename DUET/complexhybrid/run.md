# Complex Hybrid Run

## 1. Train YOLO11-L

```powershell
python DUET/complexhybrid/train_yolo11l.py
```

## 2. Train RT-DETR-L

```powershell
python DUET/complexhybrid/train_rtdetrl.py
```

## 3. Build pseudo-labeled dataset

```powershell
python DUET/complexhybrid/build_pseudo_dataset.py --unlabeled-images PATH_TO_EXTRA_IMAGES
```

## 4. Retrain on pseudo labels

```powershell
python DUET/complexhybrid/retrain_detector.py --model yolo11l.pt --data DUET/complexhybrid/pseudo_dataset/data.yaml
python DUET/complexhybrid/retrain_detector.py --model rtdetr-l.pt --data DUET/complexhybrid/pseudo_dataset/data.yaml
```

## 5. Generate submission

```powershell
python DUET/complexhybrid/generate_submission.py
```

## Notes

- The scripts default to the best RoadVision split.
- If pseudo-labeled weights are not present, the submission generator still works with the base detector weights you point it to.
