# Hybrid Run Notes

## 1. Train the detector

```powershell
python DUET/YOLO11/train_yolo.py --model yolo26m.pt --data DUET/RoadVision_DUET/Modified_YOLO_Effective/data_train_all.yaml --name roadvision_yolo26m_modified_all
```

## 2. Train the CNN crop classifier

```powershell
python DUET/RoadVision_DUET/CNN_Crop_Classifier/train_cnn.py --model convnext_tiny --epochs 30 --batch 32 --output DUET/RoadVision_DUET/CNN_Crop_Classifier/runs/convnext_tiny
```

## 3. Hybrid inference

A runnable hybrid inference script should:

- load `weights/best.pt` from YOLO26m
- load the CNN checkpoint from the crop classifier run folder
- run YOLO on each test image
- crop each detected object
- classify each crop with the CNN
- format and save the Kaggle CSV

## 4. Output folder suggestion

Use a dedicated run folder such as:

- `DUET/hybrid/runs/yolo26m_cnn_hybrid`

This keeps the hybrid artifacts separate from the detector and CNN training outputs.
