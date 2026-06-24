# DUET Hybrid Pipeline

This folder documents the two-stage hybrid inference flow for RoadVision DUET:

1. YOLO26m detects vehicles and produces bounding boxes.
2. The detected crops are passed to the CNN crop classifier.
3. The CNN refines the class label.
4. The final prediction is written to a Kaggle submission CSV.

## Inputs

- YOLO detector weights: `DUET/YOLO11/runs/roadvision_yolo26m_modified_all/weights/best.pt`
- CNN classifier weights: `DUET/RoadVision_DUET/CNN_Crop_Classifier/runs/convnext_tiny/best.pt`
- Test images: `DUET/RoadVision_DUET/test/images`

## Recommended Training Setup

- Detector training data: `DUET/RoadVision_DUET/Modified_YOLO_Effective/data_train_all.yaml`
- CNN training data: `DUET/RoadVision_DUET/CNN_Crop_Classifier/dataset/crops`

## Notes

- This folder is documentation only for now.
- The existing CNN classifier is trainable from `train_cnn.py` and uses the same 13 RoadVision classes as YOLO.
- If you want, the next step is to add a runnable hybrid inference script here.
