# YOLO + CNN Crop Classifier

This folder builds the second-stage CNN classifier for a hybrid detector:

1. YOLO localizes vehicles.
2. Vehicle crops are passed to a pretrained CNN.
3. The CNN refines the class label for visually confusing vehicle types.

## Build Crop Dataset

Run from `DUET/RoadVision_DUET/CNN_Crop_Classifier`:

```powershell
python build_cnn_crop_dataset.py
```

Output:

- `dataset/crops/train/<class_folder>/*.jpg`
- `dataset/crops/val/<class_folder>/*.jpg`
- `dataset/reports/crop_index.csv`
- `dataset/reports/crop_dataset_summary.json`

Low-count training classes are augmented from clear source images only.

## Train CNN

```powershell
python train_cnn.py --model convnext_tiny --epochs 30 --batch 32
```

Other supported first baselines:

```powershell
python train_cnn.py --model efficientnet_b3 --epochs 35 --batch 24
python train_cnn.py --model resnet50 --epochs 30 --batch 32
```

Use the CNN after YOLO prediction to refine class IDs while keeping YOLO's boxes.
