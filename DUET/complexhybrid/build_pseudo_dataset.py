from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
ROADVISION_DIR = PROJECT_DIR / "DUET" / "RoadVision_DUET"
SOURCE_DATA = ROADVISION_DIR / "Modified_YOLO_Effective"
DEFAULT_OUTPUT = BASE_DIR / "pseudo_dataset"
CLASS_NAMES = [
    "Rickshaw",
    "Motorcycle",
    "Tempu",
    "Sedan Car",
    "Pickup",
    "Microbus",
    "Mini Bus",
    "Mini Truck",
    "Agro Use",
    "Medium Truck",
    "Large Bus",
    "Heavy Truck",
    "Trailer",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(src_dir: Path, dst_dir: Path, suffixes: set[str]) -> int:
    count = 0
    ensure_dir(dst_dir)
    for path in sorted(src_dir.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in suffixes:
            shutil.copy2(path, dst_dir / path.name)
            count += 1
    return count


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def write_label(label_path: Path, boxes: list[list[float]]) -> None:
    ensure_dir(label_path.parent)
    with label_path.open("w", encoding="utf-8") as f:
        for cls, x_center, y_center, width, height in boxes:
            f.write(
                " ".join(
                    [
                        str(int(cls)),
                        format_float(x_center),
                        format_float(y_center),
                        format_float(width),
                        format_float(height),
                    ]
                )
                + "\n"
            )


def predict_boxes(model: YOLO, image_paths: list[Path], device: str | int, imgsz: int, conf: float, iou: float, max_det: int) -> list:
    kwargs = dict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, batch=1, stream=True, verbose=False)
    if str(device).lower() != "cpu":
        kwargs["device"] = device
        kwargs["half"] = True
    else:
        kwargs["device"] = "cpu"
    try:
        return list(model.predict(**kwargs))
    except torch.OutOfMemoryError:
        kwargs["device"] = "cpu"
        kwargs["half"] = False
        return list(model.predict(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a pseudo-labeled detector dataset from the best RoadVision split.")
    parser.add_argument("--teacher-weights", type=Path, required=True)
    parser.add_argument("--unlabeled-images", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=0)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()

    if not args.teacher_weights.exists():
        raise FileNotFoundError(f"Teacher weights not found: {args.teacher_weights}")

    teacher = YOLO(str(args.teacher_weights))

    train_img_src = SOURCE_DATA / "images" / "train_all"
    train_lbl_src = SOURCE_DATA / "labels" / "train_all"
    val_img_src = SOURCE_DATA / "images" / "val"
    val_lbl_src = SOURCE_DATA / "labels" / "val"

    out_img_train = args.output_dir / "images" / "train"
    out_lbl_train = args.output_dir / "labels" / "train"
    out_img_val = args.output_dir / "images" / "val"
    out_lbl_val = args.output_dir / "labels" / "val"

    base_train_count = copy_tree(train_img_src, out_img_train, {".jpg", ".jpeg", ".png"})
    base_val_count = copy_tree(val_img_src, out_img_val, {".jpg", ".jpeg", ".png"})

    for label_dir, source_dir, target_dir in [
        (train_lbl_src, train_img_src, out_lbl_train),
        (val_lbl_src, val_img_src, out_lbl_val),
    ]:
        ensure_dir(target_dir)
        for label_file in sorted(label_dir.glob("*.txt"), key=lambda p: p.name):
            shutil.copy2(label_file, target_dir / label_file.name)

    pseudo_count = 0
    manifest_rows: list[dict[str, str]] = []
    if args.unlabeled_images is not None:
        image_paths = sorted([p for p in args.unlabeled_images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.name)
        if image_paths:
            results = predict_boxes(teacher, image_paths, args.device, args.imgsz, args.conf, args.iou, args.max_det)
            for image_path, result in zip(image_paths, results):
                shutil.copy2(image_path, out_img_train / image_path.name)
                boxes = []
                if result.boxes is not None and len(result.boxes) > 0:
                    cls_values = result.boxes.cls.detach().cpu().tolist()
                    xywhn_values = result.boxes.xywhn.detach().cpu().tolist()
                    conf_values = result.boxes.conf.detach().cpu().tolist()
                    for cls, conf, (x_center, y_center, width, height) in zip(cls_values, conf_values, xywhn_values):
                        if float(conf) < args.conf:
                            continue
                        boxes.append([cls, x_center, y_center, width, height])
                write_label(out_lbl_train / f"{image_path.stem}.txt", boxes)
                manifest_rows.append({"image": image_path.name, "label_count": str(len(boxes))})
                pseudo_count += 1

    ensure_dir(args.output_dir)
    with (args.output_dir / "data.yaml").open("w", encoding="utf-8") as f:
        f.write(f"path: {args.output_dir.as_posix()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("nc: 13\n")
        f.write("names:\n")
        for idx, name in enumerate(CLASS_NAMES):
            f.write(f"  {idx}: {name}\n")

    summary = {
        "source_dataset": str(SOURCE_DATA),
        "output_dataset": str(args.output_dir),
        "base_train_images": base_train_count,
        "base_val_images": base_val_count,
        "pseudo_images": pseudo_count,
        "teacher_weights": str(args.teacher_weights),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if manifest_rows:
        with (args.output_dir / "pseudo_manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "label_count"])
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Built pseudo dataset at {args.output_dir}")


if __name__ == "__main__":
    main()
