from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageEnhance, ImageOps


BASE_DIR = Path(__file__).resolve().parent
ROADVISION_DIR = BASE_DIR.parent

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

CLEAR_CONDITIONS = {"normal_day", "bright_day", "high_contrast_day"}


def safe_class_dir(class_id: int) -> str:
    return f"{class_id:02d}_{CLASS_NAMES[class_id].replace(' ', '_')}"


def read_annotations(csv_path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = Path(row["image_id"]).stem
            annotations[stem].append(
                (
                    int(row["class_id"]),
                    float(row["x_center"]),
                    float(row["y_center"]),
                    float(row["width"]),
                    float(row["height"]),
                )
            )
    return annotations


def read_split_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row["stem"]] = row
    return metadata


def yolo_to_xyxy(box: tuple[int, float, float, float, float], image_w: int, image_h: int, pad: float) -> tuple[int, int, int, int]:
    _, x, y, w, h = box
    bw = w * image_w
    bh = h * image_h
    cx = x * image_w
    cy = y * image_h
    pad_w = bw * pad
    pad_h = bh * pad
    x1 = max(0, round(cx - bw / 2 - pad_w))
    y1 = max(0, round(cy - bh / 2 - pad_h))
    x2 = min(image_w, round(cx + bw / 2 + pad_w))
    y2 = min(image_h, round(cy + bh / 2 + pad_h))
    return x1, y1, x2, y2


def is_good_crop(crop: Image.Image, min_size: int, min_area: int) -> bool:
    w, h = crop.size
    return w >= min_size and h >= min_size and (w * h) >= min_area


def augment_crop(crop: Image.Image, rng: random.Random) -> Image.Image:
    img = crop.copy()
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)
    if rng.random() < 0.75:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.20))
    if rng.random() < 0.75:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.18))
    if rng.random() < 0.75:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.25))
    if rng.random() < 0.35:
        angle = rng.uniform(-5.0, 5.0)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
    return img


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val"):
        for class_id in range(len(CLASS_NAMES)):
            (output / "crops" / split / safe_class_dir(class_id)).mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)


def write_crop_index(rows: list[dict[str, str]], output: Path) -> None:
    fieldnames = [
        "split",
        "class_id",
        "class_name",
        "crop_path",
        "source_image",
        "source_condition",
        "source_camera",
        "x1",
        "y1",
        "x2",
        "y2",
        "augmented",
    ]
    with (output / "reports" / "crop_index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    root = args.root
    output = args.output
    image_dir = root / "RealData" / "train" / "images"
    annotations = read_annotations(root / "RealData" / "train" / "train.csv")
    metadata_path = root / "Modified_YOLO_Effective" / "reports" / "image_metadata.csv"
    metadata = read_split_metadata(metadata_path)

    reset_output(output)

    crop_rows: list[dict[str, str]] = []
    train_clear_by_class: dict[int, list[Path]] = defaultdict(list)
    original_counts: Counter = Counter()
    skipped_small = 0

    for stem, boxes in sorted(annotations.items()):
        if stem not in metadata:
            raise KeyError(f"Missing split metadata for {stem}. Build Modified_YOLO_Effective first.")
        split = metadata[stem]["split"]
        image_path = image_dir / f"{stem}.jpg"
        condition = metadata[stem]["condition"]
        camera = metadata[stem]["camera"]
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_w, image_h = img.size
            for obj_idx, box in enumerate(boxes):
                class_id = box[0]
                x1, y1, x2, y2 = yolo_to_xyxy(box, image_w, image_h, args.pad)
                crop = img.crop((x1, y1, x2, y2))
                if not is_good_crop(crop, args.min_size, args.min_area):
                    skipped_small += 1
                    continue

                class_dir = output / "crops" / split / safe_class_dir(class_id)
                crop_name = f"{stem}_obj{obj_idx:03d}_c{class_id}.jpg"
                crop_path = class_dir / crop_name
                crop.save(crop_path, quality=95)
                original_counts[(split, class_id)] += 1
                if split == "train" and condition in CLEAR_CONDITIONS:
                    train_clear_by_class[class_id].append(crop_path)

                crop_rows.append(
                    {
                        "split": split,
                        "class_id": str(class_id),
                        "class_name": CLASS_NAMES[class_id],
                        "crop_path": str(crop_path.relative_to(output)),
                        "source_image": image_path.name,
                        "source_condition": condition,
                        "source_camera": camera,
                        "x1": str(x1),
                        "y1": str(y1),
                        "x2": str(x2),
                        "y2": str(y2),
                        "augmented": "0",
                    }
                )

    train_counts = [original_counts[("train", class_id)] for class_id in range(len(CLASS_NAMES))]
    nonzero_counts = [count for count in train_counts if count > 0]
    target_count = max(args.min_aug_target, int(median(nonzero_counts))) if nonzero_counts else args.min_aug_target
    target_count = min(target_count, args.max_aug_target)

    augmented_counts: Counter = Counter()
    for class_id in range(len(CLASS_NAMES)):
        current = original_counts[("train", class_id)]
        needed = max(0, target_count - current)
        source_paths = train_clear_by_class[class_id]
        if not source_paths:
            continue
        class_dir = output / "crops" / "train" / safe_class_dir(class_id)
        for aug_idx in range(needed):
            src = rng.choice(source_paths)
            with Image.open(src) as crop:
                aug = augment_crop(crop.convert("RGB"), rng)
            aug_name = f"aug_{aug_idx:04d}_{src.stem}.jpg"
            aug_path = class_dir / aug_name
            aug.save(aug_path, quality=94)
            augmented_counts[class_id] += 1
            crop_rows.append(
                {
                    "split": "train",
                    "class_id": str(class_id),
                    "class_name": CLASS_NAMES[class_id],
                    "crop_path": str(aug_path.relative_to(output)),
                    "source_image": src.name,
                    "source_condition": "clear_augmented",
                    "source_camera": "",
                    "x1": "",
                    "y1": "",
                    "x2": "",
                    "y2": "",
                    "augmented": "1",
                }
            )

    write_crop_index(crop_rows, output)
    summary = {
        "source": str(root),
        "output": str(output),
        "crop_padding_ratio": args.pad,
        "min_size": args.min_size,
        "min_area": args.min_area,
        "skipped_small_objects": skipped_small,
        "augmentation": {
            "clear_conditions": sorted(CLEAR_CONDITIONS),
            "target_train_count_per_low_class": target_count,
            "augmented_counts": {str(i): augmented_counts.get(i, 0) for i in range(len(CLASS_NAMES))},
        },
        "class_names": CLASS_NAMES,
        "original_counts": {
            "train": {str(i): original_counts.get(("train", i), 0) for i in range(len(CLASS_NAMES))},
            "val": {str(i): original_counts.get(("val", i), 0) for i in range(len(CLASS_NAMES))},
        },
        "final_train_counts": {
            str(i): original_counts.get(("train", i), 0) + augmented_counts.get(i, 0) for i in range(len(CLASS_NAMES))
        },
        "final_val_counts": {str(i): original_counts.get(("val", i), 0) for i in range(len(CLASS_NAMES))},
    }
    (output / "reports" / "crop_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CNN crop-classifier data from RoadVision YOLO boxes.")
    parser.add_argument("--root", type=Path, default=ROADVISION_DIR)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "dataset")
    parser.add_argument("--pad", type=float, default=0.12, help="Extra context around each YOLO box.")
    parser.add_argument("--min-size", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=256)
    parser.add_argument("--min-aug-target", type=int, default=250)
    parser.add_argument("--max-aug-target", type=int, default=700)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
