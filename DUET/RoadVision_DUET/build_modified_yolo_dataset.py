from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from PIL import Image, ImageStat


BASE_DIR = Path(__file__).resolve().parent


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


@dataclass(frozen=True)
class ImageInfo:
    stem: str
    file_name: str
    source: str
    camera: str
    frame: int
    block_id: str
    width: int
    height: int
    brightness: float
    contrast: float
    saturation: float
    dark_ratio: float
    bright_ratio: float
    condition: str
    boxes: int
    class_counts: dict[int, int]


def stem_from_image_id(image_id: str) -> str:
    return Path(image_id).stem


def parse_name(stem: str, block_size: int) -> tuple[str, str, int, str]:
    source, frame_text = stem.rsplit("_", 1)
    camera = source.split("^", 1)[0]
    frame = int(frame_text)
    block_id = f"{source}_block{frame // block_size:03d}"
    return source, camera, frame, block_id


def read_annotations(csv_path: Path) -> dict[str, list[tuple[int, float, float, float, float]]]:
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = stem_from_image_id(row["image_id"])
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


def classify_condition(brightness: float, contrast: float, saturation: float, dark_ratio: float) -> str:
    if brightness < 108 or dark_ratio > 0.08:
        return "dim_cctv"
    if saturation < 22 and contrast < 45:
        return "hazy_flat"
    if contrast > 58:
        return "high_contrast_day"
    if brightness > 132:
        return "bright_day"
    return "normal_day"


def image_stats(image_path: Path) -> tuple[int, int, float, float, float, float, float, str]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        sample = img.resize((160, max(1, round(160 * height / width))))
        gray = sample.convert("L")
        gray_stat = ImageStat.Stat(gray)
        brightness = float(gray_stat.mean[0])
        contrast = float(gray_stat.stddev[0])
        hist = gray.histogram()
        total = sum(hist)
        dark_ratio = sum(hist[:45]) / total
        bright_ratio = sum(hist[210:]) / total
        hsv = sample.convert("HSV")
        saturation = float(ImageStat.Stat(hsv).mean[1])
        condition = classify_condition(brightness, contrast, saturation, dark_ratio)
        return width, height, brightness, contrast, saturation, dark_ratio, bright_ratio, condition


def collect_image_info(root: Path, block_size: int) -> list[ImageInfo]:
    train_dir = root / "RealData" / "train"
    image_dir = train_dir / "images"
    annotations = read_annotations(train_dir / "train.csv")
    infos: list[ImageInfo] = []

    for image_path in sorted(image_dir.glob("*.jpg"), key=lambda p: p.name):
        source, camera, frame, block_id = parse_name(image_path.stem, block_size)
        width, height, brightness, contrast, saturation, dark_ratio, bright_ratio, condition = image_stats(image_path)
        class_counts: Counter = Counter(cls for cls, *_ in annotations.get(image_path.stem, []))
        infos.append(
            ImageInfo(
                stem=image_path.stem,
                file_name=image_path.name,
                source=source,
                camera=camera,
                frame=frame,
                block_id=block_id,
                width=width,
                height=height,
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                dark_ratio=dark_ratio,
                bright_ratio=bright_ratio,
                condition=condition,
                boxes=sum(class_counts.values()),
                class_counts=dict(class_counts),
            )
        )
    return infos


def counter_for(infos: list[ImageInfo], key: str) -> Counter:
    if key == "class":
        counts: Counter = Counter()
        for info in infos:
            counts.update(info.class_counts)
        return counts
    return Counter(getattr(info, key) for info in infos)


def split_by_blocks(infos: list[ImageInfo], val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_block: dict[str, list[ImageInfo]] = defaultdict(list)
    for info in infos:
        by_block[info.block_id].append(info)

    blocks = list(by_block)
    total_images = len(infos)
    target_images = max(1, round(total_images * val_ratio))
    total_class = counter_for(infos, "class")
    total_condition = counter_for(infos, "condition")
    total_camera = counter_for(infos, "camera")

    def split_score(candidate_infos: list[ImageInfo]) -> float:
        candidate_images = len(candidate_infos)
        size_score = abs(candidate_images - target_images) / target_images
        class_counts = counter_for(candidate_infos, "class")
        condition_counts = counter_for(candidate_infos, "condition")
        camera_counts = counter_for(candidate_infos, "camera")
        class_score = 0.0
        class_fraction_penalty = 0.0
        for cls in range(len(CLASS_NAMES)):
            total = max(1, total_class[cls])
            got_fraction = class_counts[cls] / total
            class_score += abs(got_fraction - val_ratio)
            if class_counts[cls] == 0:
                class_fraction_penalty += 0.60
            if got_fraction > 0.28:
                class_fraction_penalty += (got_fraction - 0.28) * 6.0
            if total_class[cls] >= 80 and got_fraction < 0.06:
                class_fraction_penalty += (0.06 - got_fraction) * 3.0
        condition_score = 0.0
        for condition in total_condition:
            target = total_condition[condition] / total_images
            got = condition_counts[condition] / max(1, candidate_images)
            condition_score += abs(got - target)
        camera_score = 0.0
        for camera in total_camera:
            target = total_camera[camera] / total_images
            got = camera_counts[camera] / max(1, candidate_images)
            camera_score += abs(got - target)
        missing_condition_penalty = sum(1 for c in total_condition if condition_counts[c] == 0) * 0.20
        return size_score * 1.6 + class_score * 2.5 + class_fraction_penalty + condition_score + camera_score + missing_condition_penalty

    best_selected: set[str] = set()
    best_score = math.inf
    for _ in range(25000):
        shuffled = blocks[:]
        rng.shuffle(shuffled)
        selected: set[str] = set()
        selected_infos: list[ImageInfo] = []
        for block in shuffled:
            next_size = len(selected_infos) + len(by_block[block])
            if next_size > target_images * 1.18:
                continue
            selected.add(block)
            selected_infos.extend(by_block[block])
            if len(selected_infos) >= target_images * 0.92:
                break
        if not selected_infos:
            continue
        score = split_score(selected_infos)
        if score < best_score:
            best_score = score
            best_selected = selected

    if not best_selected:
        raise RuntimeError("Could not create a validation split.")

    val_stems = {info.stem for block in best_selected for info in by_block[block]}
    train_stems = {info.stem for info in infos if info.stem not in val_stems}
    return train_stems, val_stems


def write_label(label_path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w", encoding="utf-8", newline="\n") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def reset_output(output: Path) -> None:
    if output.exists():
        shutil.rmtree(output)
    for split in ("train", "val", "train_all", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "train_all"):
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    (output / "reports").mkdir(parents=True, exist_ok=True)


def copy_dataset(root: Path, output: Path, infos: list[ImageInfo], train_stems: set[str], val_stems: set[str]) -> None:
    annotations = read_annotations(root / "RealData" / "train" / "train.csv")
    train_image_dir = root / "RealData" / "train" / "images"
    test_image_dir = root / "test" / "images"

    info_by_stem = {info.stem: info for info in infos}
    for split, stems in (("train", train_stems), ("val", val_stems), ("train_all", set(info_by_stem))):
        for stem in sorted(stems):
            src = train_image_dir / f"{stem}.jpg"
            shutil.copy2(src, output / "images" / split / src.name)
            write_label(output / "labels" / split / f"{stem}.txt", annotations.get(stem, []))

    for image_path in sorted(test_image_dir.glob("*.jpg"), key=lambda p: p.name):
        shutil.copy2(image_path, output / "images" / "test" / image_path.name)


def write_yaml(output: Path) -> None:
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    base = output.resolve().as_posix()
    (output / "data.yaml").write_text(
        f"path: {base}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 13\nnames:\n{names}\n",
        encoding="utf-8",
    )
    (output / "data_train_all.yaml").write_text(
        f"path: {base}\ntrain: images/train_all\nval: images/val\ntest: images/test\nnc: 13\nnames:\n{names}\n",
        encoding="utf-8",
    )


def write_reports(output: Path, infos: list[ImageInfo], train_stems: set[str], val_stems: set[str], block_size: int) -> None:
    fieldnames = [
        "split",
        "file_name",
        "stem",
        "source",
        "camera",
        "frame",
        "block_id",
        "width",
        "height",
        "brightness",
        "contrast",
        "saturation",
        "dark_ratio",
        "bright_ratio",
        "condition",
        "boxes",
    ] + [f"class_{i}" for i in range(len(CLASS_NAMES))]

    with (output / "reports" / "image_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for info in infos:
            split = "val" if info.stem in val_stems else "train"
            row = {
                "split": split,
                "file_name": info.file_name,
                "stem": info.stem,
                "source": info.source,
                "camera": info.camera,
                "frame": info.frame,
                "block_id": info.block_id,
                "width": info.width,
                "height": info.height,
                "brightness": round(info.brightness, 3),
                "contrast": round(info.contrast, 3),
                "saturation": round(info.saturation, 3),
                "dark_ratio": round(info.dark_ratio, 5),
                "bright_ratio": round(info.bright_ratio, 5),
                "condition": info.condition,
                "boxes": info.boxes,
            }
            row.update({f"class_{i}": info.class_counts.get(i, 0) for i in range(len(CLASS_NAMES))})
            writer.writerow(row)

    split_infos = {
        "train": [info for info in infos if info.stem in train_stems],
        "val": [info for info in infos if info.stem in val_stems],
        "all": infos,
    }
    summary = {
        "strategy": "contiguous frame-block split with class and lighting balance",
        "block_size_frames": block_size,
        "class_names": CLASS_NAMES,
        "splits": {},
    }
    for split, split_info in split_infos.items():
        class_counts = counter_for(split_info, "class")
        condition_counts = counter_for(split_info, "condition")
        camera_counts = counter_for(split_info, "camera")
        brightness_values = [info.brightness for info in split_info]
        summary["splits"][split] = {
            "images": len(split_info),
            "boxes": sum(class_counts.values()),
            "class_counts": {str(i): class_counts.get(i, 0) for i in range(len(CLASS_NAMES))},
            "condition_counts": dict(sorted(condition_counts.items())),
            "camera_counts": dict(sorted(camera_counts.items())),
            "brightness_mean": round(mean(brightness_values), 3) if brightness_values else 0,
        }
    (output / "reports" / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        """# Modified RoadVision YOLO Dataset

This is a YOLO-ready rearranged dataset built from `RealData/train` and `test/images`.

## Why This Split

- Uses contiguous frame blocks instead of a pure random split.
- Balances validation using class counts and image lighting statistics.
- Keeps metadata reports for brightness, contrast, saturation, camera, sequence, frame, and condition.

## Train

```powershell
yolo detect train data=DUET/RoadVision_DUET/Modified_YOLO_Effective/data.yaml model=yolo11m.pt imgsz=800 batch=4 epochs=100 device=0 multi_scale=False
```

## Final Full-Data Training

After choosing settings from validation, train on all 810 labeled images:

```powershell
yolo detect train data=DUET/RoadVision_DUET/Modified_YOLO_Effective/data_train_all.yaml model=yolo11m.pt imgsz=800 batch=4 epochs=100 device=0 multi_scale=False
```

Reports live in `reports/`.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a modified, analyzed YOLO dataset for RoadVision DUET.")
    parser.add_argument("--root", type=Path, default=BASE_DIR)
    parser.add_argument("--output-name", default="Modified_YOLO_Effective")
    parser.add_argument("--val-ratio", type=float, default=0.17)
    parser.add_argument("--block-size", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = args.root / args.output_name
    infos = collect_image_info(args.root, args.block_size)
    train_stems, val_stems = split_by_blocks(infos, args.val_ratio, args.seed)

    reset_output(output)
    copy_dataset(args.root, output, infos, train_stems, val_stems)
    write_yaml(output)
    write_reports(output, infos, train_stems, val_stems, args.block_size)
    write_readme(output)

    summary = json.loads((output / "reports" / "split_summary.json").read_text(encoding="utf-8"))
    print(json.dumps(summary, indent=2))
    print(f"\nCreated modified dataset: {output}")


if __name__ == "__main__":
    main()
