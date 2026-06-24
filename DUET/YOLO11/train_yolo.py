from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an Ultralytics YOLO baseline on RoadVision DUET.")
    parser.add_argument("--data", type=Path, default=BASE_DIR.parent / "RoadVision_DUET" / "Modified_YOLO_Effective" / "data_train_all.yaml")
    parser.add_argument("--model", default="yolo26s.pt", help="Examples: yolo11s.pt, yolo11m.pt, yolo26m.pt, yolo26x.pt, or a URL/path.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=1240)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=BASE_DIR / "runs")
    parser.add_argument("--name", default="roadvision_yolo26s_modified_all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        optimizer="auto",
        cos_lr=True,
        close_mosaic=20,
        cache="disk",
        multi_scale=False,
        hsv_h=0.015,
        hsv_s=0.65,
        hsv_v=0.45,
        degrees=3.0,
        translate=0.12,
        scale=0.60,
        shear=1.5,
        perspective=0.0005,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.12,
        copy_paste=0.10,
    )


if __name__ == "__main__":
    main()
