from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
ROADVISION_DIR = PROJECT_DIR / "DUET" / "RoadVision_DUET"
DEFAULT_YOLO_WEIGHTS = BASE_DIR / "runs" / "yolo11l_modified_all" / "weights" / "best.pt"
DEFAULT_RTDETR_WEIGHTS = BASE_DIR / "runs" / "rtdetrl_modified_all" / "weights" / "best.pt"
DEFAULT_TEST_IMAGES = ROADVISION_DIR / "test" / "images"
DEFAULT_SAMPLE = ROADVISION_DIR / "test" / "sample_submission.csv"
DEFAULT_OUTPUT = BASE_DIR / "submission.csv"


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def run_prediction(model: YOLO, image_paths: list[Path], device: str | int, imgsz: int, conf: float, iou: float, max_det: int, batch: int):
    kwargs = dict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, batch=batch, stream=True, verbose=False)
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


def to_normalized_boxes(result, width: int, height: int, model_index: int) -> list[dict[str, float]]:
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes

    cls_values = result.boxes.cls.detach().cpu().tolist()
    conf_values = result.boxes.conf.detach().cpu().tolist()
    xyxy_values = result.boxes.xyxy.detach().cpu().tolist()
    for cls, conf, (left, top, right, bottom) in zip(cls_values, conf_values, xyxy_values):
        x1 = max(0.0, min(1.0, float(left) / width))
        y1 = max(0.0, min(1.0, float(top) / height))
        x2 = max(0.0, min(1.0, float(right) / width))
        y2 = max(0.0, min(1.0, float(bottom) / height))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"cls": float(int(cls)), "score": float(conf), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "model_index": float(model_index)})
    return boxes


def iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    inter_x1 = max(box_a["x1"], box_b["x1"])
    inter_y1 = max(box_a["y1"], box_b["y1"])
    inter_x2 = min(box_a["x2"], box_b["x2"])
    inter_y2 = min(box_a["y2"], box_b["y2"])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, box_a["x2"] - box_a["x1"]) * max(0.0, box_a["y2"] - box_a["y1"])
    area_b = max(0.0, box_b["x2"] - box_b["x1"]) * max(0.0, box_b["y2"] - box_b["y1"])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def weighted_box_fusion(box_sets: list[list[dict[str, float]]], iou_thr: float, model_weights: list[float]) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for boxes in box_sets:
        candidates.extend(boxes)
    if not candidates:
        return []

    candidates.sort(key=lambda b: b["score"], reverse=True)
    clusters: list[list[dict[str, float]]] = []

    for box in candidates:
        matched_cluster = None
        for cluster in clusters:
            if int(cluster[0]["cls"]) != int(box["cls"]):
                continue
            if any(iou(cluster_box, box) >= iou_thr for cluster_box in cluster):
                matched_cluster = cluster
                break
        if matched_cluster is None:
            clusters.append([box])
        else:
            matched_cluster.append(box)

    fused: list[dict[str, float]] = []
    for cluster in clusters:
        weight_total = 0.0
        x1 = y1 = x2 = y2 = score_total = 0.0
        for box in cluster:
            weight = box["score"] * model_weights[int(box["model_index"])]
            weight_total += weight
            x1 += box["x1"] * weight
            y1 += box["y1"] * weight
            x2 += box["x2"] * weight
            y2 += box["y2"] * weight
            score_total += box["score"] * weight
        if weight_total <= 0:
            continue
        fused.append(
            {
                "cls": float(int(cluster[0]["cls"])),
                "score": min(1.0, score_total / weight_total),
                "x1": max(0.0, min(1.0, x1 / weight_total)),
                "y1": max(0.0, min(1.0, y1 / weight_total)),
                "x2": max(0.0, min(1.0, x2 / weight_total)),
                "y2": max(0.0, min(1.0, y2 / weight_total)),
            }
        )

    fused.sort(key=lambda b: b["score"], reverse=True)
    return fused


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RoadVision submission.csv from YOLO11-L + RT-DETR-L with WBF.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--rtdetr-weights", type=Path, default=DEFAULT_RTDETR_WEIGHTS)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--wbf-iou", type=float, default=0.55)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.test_images.exists():
        raise FileNotFoundError(f"Test image folder not found: {args.test_images}")

    image_map = {p.stem: p for p in sorted(args.test_images.iterdir(), key=lambda p: p.name) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if args.sample.exists():
        with args.sample.open("r", newline="", encoding="utf-8") as f:
            sample_rows = list(csv.DictReader(f))
        ordered_ids = [row["image_id"] for row in sample_rows]
    else:
        ordered_ids = sorted(image_map.keys())

    ordered_paths = []
    for image_id in ordered_ids:
        if image_id not in image_map:
            raise FileNotFoundError(f"Missing test image for image_id: {image_id}")
        ordered_paths.append(image_map[image_id])

    device = args.device
    yolo_model = YOLO(str(args.yolo_weights))
    yolo_results = run_prediction(yolo_model, ordered_paths, device, args.imgsz, args.conf, args.iou, args.max_det, args.batch)

    rtdetr_results = []
    if args.rtdetr_weights.exists():
        rtdetr_model = YOLO(str(args.rtdetr_weights))
        rtdetr_results = run_prediction(rtdetr_model, ordered_paths, device, args.imgsz, args.conf, args.iou, args.max_det, args.batch)
        print(f"Loaded RT-DETR weights from {args.rtdetr_weights}")
    else:
        print(f"RT-DETR weights not found at {args.rtdetr_weights}; using YOLO only.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "PredictionString"])
        writer.writeheader()

        for index, image_path in enumerate(ordered_paths):
            width = height = 1
            try:
                from PIL import Image

                width, height = Image.open(image_path).size
            except Exception:
                pass

            box_sets = [to_normalized_boxes(yolo_results[index], width, height, 0)]
            if rtdetr_results:
                box_sets.append(to_normalized_boxes(rtdetr_results[index], width, height, 1))

            fused_boxes = weighted_box_fusion(box_sets, args.wbf_iou, model_weights=[1.0, 1.0][: len(box_sets)])
            parts: list[str] = []
            for box in fused_boxes:
                x_center = (box["x1"] + box["x2"]) / 2.0
                y_center = (box["y1"] + box["y2"]) / 2.0
                box_w = box["x2"] - box["x1"]
                box_h = box["y2"] - box["y1"]
                parts.extend(
                    [
                        str(int(box["cls"])),
                        format_float(box["score"]),
                        format_float(x_center),
                        format_float(y_center),
                        format_float(box_w),
                        format_float(box_h),
                    ]
                )

            writer.writerow({"image_id": image_path.stem, "PredictionString": " ".join(parts)})

    print(f"Wrote {len(ordered_paths)} rows to {args.output}")


if __name__ == "__main__":
    main()
