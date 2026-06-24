from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "pseudo_dataset" / "data.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain a detector on the pseudo-labeled RoadVision dataset.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="yolo11l.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=BASE_DIR / "runs")
    parser.add_argument("--name", default="pseudo_retrain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=35)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {args.data}")

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
        optimizer="AdamW",
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
