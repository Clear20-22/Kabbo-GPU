from __future__ import annotations

import argparse
import csv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
ROADVISION_DIR = PROJECT_DIR / "DUET" / "RoadVision_DUET"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RoadVision DUET submission format.")
    parser.add_argument("--submission", type=Path, default=BASE_DIR / "submission.csv")
    parser.add_argument("--test-images", type=Path, default=ROADVISION_DIR / "test" / "images")
    args = parser.parse_args()

    expected_ids = [p.stem for p in sorted(args.test_images.glob("*.jpg"), key=lambda p: p.name)]
    expected_set = set(expected_ids)
    rows = list(csv.DictReader(args.submission.open("r", newline="", encoding="utf-8")))
    ids = [row["image_id"] for row in rows]
    id_set = set(ids)

    errors: list[str] = []
    if list(csv.DictReader(args.submission.open("r", newline="", encoding="utf-8")).fieldnames or []) != [
        "image_id",
        "PredictionString",
    ]:
        errors.append("Header must be exactly: image_id,PredictionString")
    if len(rows) != len(expected_ids):
        errors.append(f"Row count mismatch: got {len(rows)}, expected {len(expected_ids)}")
    if len(ids) != len(id_set):
        errors.append("Duplicate image_id values found")
    missing = sorted(expected_set - id_set)
    extra = sorted(id_set - expected_set)
    if missing:
        errors.append(f"Missing expected IDs: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if extra:
        errors.append(f"Unexpected IDs: {extra[:5]}{' ...' if len(extra) > 5 else ''}")

    bad_prediction_rows = 0
    bad_ranges = 0
    detections = 0
    for row in rows:
        parts = (row.get("PredictionString") or "").strip().split()
        if not parts:
            continue
        if len(parts) % 6 != 0:
            bad_prediction_rows += 1
            continue
        for i in range(0, len(parts), 6):
            cls = int(float(parts[i]))
            conf = float(parts[i + 1])
            coords = [float(v) for v in parts[i + 2 : i + 6]]
            detections += 1
            if cls < 0 or cls > 12 or conf < 0 or conf > 1 or any(v < 0 or v > 1 for v in coords):
                bad_ranges += 1

    if bad_prediction_rows:
        errors.append(f"{bad_prediction_rows} rows have PredictionString length not divisible by 6")
    if bad_ranges:
        errors.append(f"{bad_ranges} detections have class/confidence/box values outside allowed ranges")

    print(f"submission: {args.submission}")
    print(f"rows: {len(rows)}")
    print(f"expected rows: {len(expected_ids)}")
    print(f"detections: {detections}")
    if errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()
