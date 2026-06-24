from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DEFAULT_SEARCH_DIRS = [BASE_DIR, PROJECT_DIR / "DUET" / "RoadVision_DUET" / "submissions"]
CLASS_NAMES = [
    "Rickshaw", "Motorcycle", "Tempu", "Sedan Car", "Pickup",
    "Microbus", "Mini Bus", "Mini Truck", "Agro Use",
    "Medium Truck", "Large Bus", "Heavy Truck", "Trailer",
]

@dataclass
class CsvStats:
    path: Path
    schema: str
    row_count: int
    image_ids: list[str]
    detections: int
    empty_prediction_rows: int
    malformed_prediction_rows: int
    class_counts: Counter
    class_confidences: dict[int, list[float]]
    widths: list[float]
    heights: list[float]
    areas: list[float]
    zero_area_count: int  # <-- Added to track hidden broken boxes

def resolve_csv(path_text: str) -> Path:
    candidate = Path(path_text)
    if candidate.exists(): return candidate

    for search_dir in DEFAULT_SEARCH_DIRS:
        direct = search_dir / path_text
        if direct.exists(): return direct
        if direct.suffix.lower() != ".csv":
            with_suffix = direct.with_suffix(".csv")
            if with_suffix.exists(): return with_suffix
        if candidate.suffix.lower() != ".csv":
            suffixed = search_dir / f"{path_text}.csv"
            if suffixed.exists(): return suffixed

    raise FileNotFoundError(f"Could not resolve CSV file: {path_text}")

def parse_prediction_string(prediction: str) -> tuple[list[tuple[int, float, float, float, float, float]], bool]:
    parts = prediction.strip().split()
    if not parts: return [], False
    if len(parts) % 6 != 0: return [], True

    detections: list[tuple[int, float, float, float, float, float]] = []
    for i in range(0, len(parts), 6):
        cls = int(float(parts[i]))
        conf = float(parts[i + 1])
        x_center = float(parts[i + 2])
        y_center = float(parts[i + 3])
        width = float(parts[i + 4])
        height = float(parts[i + 5])
        detections.append((cls, conf, x_center, y_center, width, height))
    return detections, False

def load_csv(path: Path) -> CsvStats:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    has_prediction_string = "PredictionString" in fieldnames
    has_flat_detection_columns = {"class_id", "x_center", "y_center", "width", "height"}.issubset(set(fieldnames))
    
    if has_prediction_string: schema = "submission"
    elif has_flat_detection_columns: schema = "detection_rows"
    else: schema = "unknown"

    image_ids: list[str] = []
    class_counts: Counter = Counter()
    class_confidences: dict[int, list[float]] = defaultdict(list)
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    detections = 0
    empty_prediction_rows = 0
    malformed_prediction_rows = 0
    zero_area_count = 0

    for row in rows:
        image_id = row.get("image_id", row.get("ImageID", "")).strip()
        image_ids.append(image_id)

        if has_prediction_string:
            prediction = (row.get("PredictionString") or "").strip()
            if not prediction:
                empty_prediction_rows += 1
                continue

            parsed, malformed = parse_prediction_string(prediction)
            if malformed:
                malformed_prediction_rows += 1
                continue

            for cls, conf, x_center, y_center, width, height in parsed:
                detections += 1
                class_counts[cls] += 1
                class_confidences[cls].append(conf)
                widths.append(width)
                heights.append(height)
                
                area = width * height
                areas.append(area)
                if area <= 0.000001:  # Enforce tiny threshold catch
                    zero_area_count += 1

        elif has_flat_detection_columns:
            try:
                cls = int(float(row["class_id"]))
                width = float(row["width"])
                height = float(row["height"])
                conf = float(row.get("confidence", row.get("Confidence", 1.0)))
            except (KeyError, TypeError, ValueError):
                malformed_prediction_rows += 1
                continue

            detections += 1
            class_counts[cls] += 1
            class_confidences[cls].append(conf)
            widths.append(width)
            heights.append(height)
            
            area = width * height
            areas.append(area)
            if area <= 0.000001:
                zero_area_count += 1
        else:
            empty_prediction_rows += 1

    return CsvStats(
        path=path, schema=schema, row_count=len(rows), image_ids=image_ids,
        detections=detections, empty_prediction_rows=empty_prediction_rows,
        malformed_prediction_rows=malformed_prediction_rows, class_counts=class_counts,
        class_confidences=class_confidences, widths=widths, heights=heights, areas=areas,
        zero_area_count=zero_area_count
    )

def stat_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values: return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
        "median": median(values),
    }

def print_model_stats(label: str, stats: CsvStats) -> None:
    print(f"\n[{label}]")
    print(f"path: {stats.path}")
    print(f"schema: {stats.schema}")
    print(f"rows: {stats.row_count}")
    print(f"image_ids: {len(stats.image_ids)}")
    print(f"detections: {stats.detections}")
    print(f"empty prediction rows: {stats.empty_prediction_rows}")
    print(f"malformed prediction rows: {stats.malformed_prediction_rows}")

    if stats.zero_area_count > 0:
        print(f"🚨 WARNING: Found {stats.zero_area_count} broken boxes with ~0.0 area!")

    if stats.detections:
        print("class counts:")
        for cls_id in range(len(CLASS_NAMES)):
            count = stats.class_counts.get(cls_id, 0)
            confs = stats.class_confidences.get(cls_id, [])
            avg_conf = mean(confs) if confs else None
            cls_name = CLASS_NAMES[cls_id]
            conf_text = f"{avg_conf:.4f}" if avg_conf is not None else "n/a"
            print(f"  {cls_id:02d} {cls_name}: {count} detections, avg_conf={conf_text}")

        print("box width stats:", json.dumps(stat_summary(stats.widths), indent=2))
        print("box height stats:", json.dumps(stat_summary(stats.heights), indent=2))
        print("box area stats:", json.dumps(stat_summary(stats.areas), indent=2))

def compare(stats_a: CsvStats, stats_b: CsvStats) -> None:
    ids_a = set(stats_a.image_ids)
    ids_b = set(stats_b.image_ids)
    overlap = ids_a & ids_b

    print("\n[Comparison]")
    print(f"shared image_ids: {len(overlap)}")
    print(f"only in first: {len(ids_a - ids_b)}")
    print(f"only in second: {len(ids_b - ids_a)}")
    print(f"row delta: {stats_a.row_count - stats_b.row_count}")
    print(f"detection delta: {stats_a.detections - stats_b.detections}")

    print("\nclass delta (first - second):")
    for cls_id in range(len(CLASS_NAMES)):
        delta = stats_a.class_counts.get(cls_id, 0) - stats_b.class_counts.get(cls_id, 0)
        print(f"  {cls_id:02d} {CLASS_NAMES[cls_id]}: {delta}")

    if overlap:
        print(f"\nshared IDs sample: {sorted(list(overlap))[:5]}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two RoadVision DUET CSV files safely.")
    parser.add_argument("csv_a", help="First CSV file name or path")
    parser.add_argument("csv_b", help="Second CSV file name or path")
    args = parser.parse_args()

    stats_a = load_csv(resolve_csv(args.csv_a))
    stats_b = load_csv(resolve_csv(args.csv_b))

    print_model_stats("First CSV", stats_a)
    print_model_stats("Second CSV", stats_b)
    compare(stats_a, stats_b)

if __name__ == "__main__":
    main()