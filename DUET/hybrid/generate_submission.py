from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
ROADVISION_DIR = PROJECT_DIR / "DUET" / "RoadVision_DUET"
# DEFAULT_YOLO_WEIGHTS = PROJECT_DIR / "DUET" / "YOLO11" / "runs" / "yolo11m_800" / "weights" / "best.pt"
DEFAULT_YOLO_WEIGHTS = PROJECT_DIR /  "runs" / "detect" / "yolo11m_heavy_traffic_finetune-4" / "weights" / "best.pt"
DEFAULT_CNN_WEIGHTS = PROJECT_DIR / "DUET" / "hybrid" / "runs" / "convnext_tinny" / "best.pt"
DEFAULT_TEST_IMAGES = ROADVISION_DIR / "test" / "images"
DEFAULT_OUTPUT = BASE_DIR / "submission.csv"


def format_float(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def build_model(model_name: str, num_classes: int):
    if model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported CNN model: {model_name}")


def build_transform(imgsz: int):
    return transforms.Compose(
        [
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


@dataclass
class CnnBundle:
    model: torch.nn.Module
    classes: list[str]
    transform: transforms.Compose


def load_cnn_bundle(weights_path: Path, device: torch.device, imgsz: int) -> CnnBundle | None:
    if not weights_path.exists():
        return None

    checkpoint = torch.load(weights_path, map_location=device)
    if not isinstance(checkpoint, dict):
        return None
    if "model" not in checkpoint or "classes" not in checkpoint or "model_name" not in checkpoint:
        return None

    classes = [str(cls) for cls in checkpoint["classes"]]
    model = build_model(str(checkpoint["model_name"]), len(classes))
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return CnnBundle(model=model, classes=classes, transform=build_transform(imgsz))


def clamp_box(left: float, top: float, right: float, bottom: float, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width, int(round(left))))
    y1 = max(0, min(height, int(round(top))))
    x2 = max(0, min(width, int(round(right))))
    y2 = max(0, min(height, int(round(bottom))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def run_cnn(bundle: CnnBundle, crop: Image.Image, device: torch.device) -> tuple[int, float]:
    with torch.no_grad():
        tensor = bundle.transform(crop).unsqueeze(0).to(device)
        logits = bundle.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        score, index = torch.max(probabilities, dim=0)
        return int(index.item()), float(score.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a hybrid RoadVision submission using YOLO crops and an optional CNN classifier.")
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument("--cnn-weights", type=Path, default=DEFAULT_CNN_WEIGHTS)
    parser.add_argument("--test-images", type=Path, default=DEFAULT_TEST_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample", type=Path, default=ROADVISION_DIR / "test" / "sample_submission.csv")
    parser.add_argument("--yolo-imgsz", type=int, default=850)
    parser.add_argument("--cnn-imgsz", type=int, default=224)
    parser.add_argument("--conf", type=float, default=0.0005)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=500)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    if not args.test_images.exists():
        raise FileNotFoundError(f"Test image folder not found: {args.test_images}")

    test_files = sorted([p for p in args.test_images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}], key=lambda p: p.name)
    if not test_files:
        raise FileNotFoundError(f"No test images found in: {args.test_images}")

    device = torch.device("cpu")
    use_cuda = str(args.device).lower() != "cpu" and torch.cuda.is_available()
    if use_cuda:
        device = torch.device(f"cuda:{args.device}") if str(args.device).isdigit() else torch.device("cuda")

    cnn_bundle = load_cnn_bundle(args.cnn_weights, device, args.cnn_imgsz)
    if cnn_bundle is None:
        print(f"CNN checkpoint not available or invalid at {args.cnn_weights}; falling back to YOLO-only labels.")
    else:
        print(f"Loaded CNN checkpoint from {args.cnn_weights}.")

    model = YOLO(str(args.yolo_weights))
    predict_kwargs = dict(
        source=[str(p) for p in test_files],
        imgsz=args.yolo_imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        batch=args.batch,
        stream=True,
        verbose=False,
    )
    if use_cuda:
        predict_kwargs["device"] = args.device
        predict_kwargs["half"] = True
    else:
        predict_kwargs["device"] = "cpu"

    try:
        results = model.predict(**predict_kwargs)
    except torch.OutOfMemoryError:
        if use_cuda:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        predict_kwargs["device"] = "cpu"
        predict_kwargs["half"] = False
        results = model.predict(**predict_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "PredictionString"])
        writer.writeheader()

        for image_path, result in zip(test_files, results):
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            parts: list[str] = []

            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                cls_values = boxes.cls.detach().cpu().tolist()
                conf_values = boxes.conf.detach().cpu().tolist()
                xyxy_values = boxes.xyxy.detach().cpu().tolist()
                order = sorted(range(len(conf_values)), key=lambda i: conf_values[i], reverse=True)

                for index in order:
                    left, top, right, bottom = xyxy_values[index]
                    clipped = clamp_box(left, top, right, bottom, width, height)
                    if clipped is None:
                        continue

                    x1, y1, x2, y2 = clipped
                    crop = image.crop((x1, y1, x2, y2))
                    yolo_cls = int(cls_values[index])
                    yolo_conf = max(0.0, min(1.0, float(conf_values[index])))
                    final_cls = yolo_cls

                    if cnn_bundle is not None:
                        try:
                            cnn_cls, _ = run_cnn(cnn_bundle, crop, device)
                            final_cls = cnn_cls
                        except Exception as exc:
                            print(f"CNN inference failed for {image_path.name}; using YOLO label. Reason: {exc}")

                    x_center = ((x1 + x2) / 2.0) / width
                    y_center = ((y1 + y2) / 2.0) / height
                    box_w = (x2 - x1) / width
                    box_h = (y2 - y1) / height
                    parts.extend(
                        [
                            str(final_cls),
                            format_float(yolo_conf),
                            format_float(x_center),
                            format_float(y_center),
                            format_float(box_w),
                            format_float(box_h),
                        ]
                    )

            writer.writerow({"image_id": image_path.stem, "PredictionString": " ".join(parts)})

    print(f"Wrote {len(test_files)} rows to {args.output}")


if __name__ == "__main__":
    main()