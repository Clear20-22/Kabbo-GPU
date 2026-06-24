from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


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


def image_stem_from_id(image_id: str) -> str:
    return Path(image_id).stem


def find_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def read_annotations(csv_path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"image_id", "class_id", "x_center", "y_center", "width", "height"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing train.csv columns: {sorted(missing)}")

        for row in reader:
            stem = image_stem_from_id(row["image_id"])
            cls = int(row["class_id"])
            box = (
                cls,
                float(row["x_center"]),
                float(row["y_center"]),
                float(row["width"]),
                float(row["height"]),
            )
            annotations[stem].append(box)
    return annotations


def split_images(
    image_paths: list[Path],
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = image_paths[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio))
    val = sorted(shuffled[:val_count], key=lambda p: p.name)
    train = sorted(shuffled[val_count:], key=lambda p: p.name)
    return train, val


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)


def copy_split(
    split: str,
    image_paths: list[Path],
    annotations: dict[str, list[tuple[int, float, float, float, float]]],
    output_dir: Path,
) -> Counter:
    image_out = output_dir / "images" / split
    label_out = output_dir / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    for image_path in image_paths:
        shutil.copy2(image_path, image_out / image_path.name)
        label_path = label_out / f"{image_path.stem}.txt"
        rows = annotations.get(image_path.stem, [])
        with label_path.open("w", encoding="utf-8", newline="\n") as f:
            for cls, x, y, w, h in rows:
                class_counts[cls] += 1
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    return class_counts


def copy_test_images(test_image_dir: Path, output_dir: Path) -> int:
    test_out = output_dir / "images" / "test"
    test_out.mkdir(parents=True, exist_ok=True)
    test_images = sorted(
        [p for p in test_image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    for image_path in test_images:
        shutil.copy2(image_path, test_out / image_path.name)
    return len(test_images)


def write_data_yaml(output_dir: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 13\n"
        "names:\n"
        f"{names}\n"
    )
    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RoadVision DUET data in Ultralytics YOLO format.")
    parser.add_argument("--root", type=Path, default=Path("DUET/RoadVision_DUET"))
    parser.add_argument("--output", type=Path, default=Path("DUET/YOLO11/yolo_dataset"))
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    train_dir = args.root / "RealData" / "train"
    train_image_dir = train_dir / "images"
    train_csv = train_dir / "train.csv"
    test_image_dir = args.root / "test" / "images"

    if not args.keep_existing:
        clean_output_dir(args.output)

    annotations = read_annotations(train_csv)
    image_paths = sorted(
        [p for p in train_image_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    missing_images = sorted(stem for stem in annotations if find_image(train_image_dir, stem) is None)
    if missing_images:
        raise FileNotFoundError(f"{len(missing_images)} annotated images are missing, first: {missing_images[:3]}")

    train_images, val_images = split_images(image_paths, args.val_ratio, args.seed)
    train_counts = copy_split("train", train_images, annotations, args.output)
    val_counts = copy_split("val", val_images, annotations, args.output)
    test_count = copy_test_images(test_image_dir, args.output)
    write_data_yaml(args.output)

    stats = {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "test_images": test_count,
        "train_boxes": sum(train_counts.values()),
        "val_boxes": sum(val_counts.values()),
        "class_names": CLASS_NAMES,
        "train_class_counts": {str(k): train_counts.get(k, 0) for k in range(len(CLASS_NAMES))},
        "val_class_counts": {str(k): val_counts.get(k, 0) for k in range(len(CLASS_NAMES))},
    }
    (args.output / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"\nYOLO data config: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
