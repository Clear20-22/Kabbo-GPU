from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_WEIGHTS = PROJECT_DIR / "runs" / "face_mask_yolo" / "weights" / "best.pt"
DEFAULT_CNN_WEIGHTS = PROJECT_DIR / "efficientnetb0_mask_status.pth"
DEFAULT_SAVE_DIR = PROJECT_DIR / "hybrid_results"
DEFAULT_CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]

try:
	LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow fallback
	LANCZOS = Image.LANCZOS


def load_yolo_class():
	try:
		return importlib.import_module("ultralytics").YOLO
	except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
		raise ModuleNotFoundError(
			"ultralytics is not installed. Install it first with: pip install ultralytics"
		) from exc


def resolve_device(device_value: str | int) -> torch.device:
	if isinstance(device_value, int):
		return torch.device(f"cuda:{device_value}") if torch.cuda.is_available() else torch.device("cpu")

	value = str(device_value).strip().lower()
	if value == "cpu":
		return torch.device("cpu")
	if value.startswith("cuda") and torch.cuda.is_available():
		if ":" in value:
			return torch.device(value)
		return torch.device("cuda:0")
	return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_cnn_model(num_classes: int):
	model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
	in_features = model.classifier[1].in_features
	model.classifier[1] = torch.nn.Sequential(
		torch.nn.Dropout(p=0.35),
		torch.nn.Linear(in_features, num_classes),
	)
	return model


def load_cnn_model(weights_path: Path, device: torch.device):
	if not weights_path.exists():
		raise FileNotFoundError(f"CNN weights file not found: {weights_path}")

	checkpoint = torch.load(weights_path, map_location=device)
	class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
	image_size = int(checkpoint.get("image_size", 320))

	model = build_cnn_model(len(class_names)).to(device)
	model.load_state_dict(checkpoint["model_state"])
	model.eval()
	return model, class_names, image_size


def build_cnn_transform(image_size: int):
	return transforms.Compose(
		[
			transforms.Lambda(lambda image: image.convert("RGB")),
			transforms.Lambda(lambda image: ImageOps.pad(image, (image_size, image_size), method=LANCZOS, color=(114, 114, 114), centering=(0.5, 0.5))),
			transforms.ToTensor(),
			transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
		]
	)


def load_image(source: Path) -> np.ndarray:
	image = cv2.imread(str(source))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {source}")
	return image


def crop_with_padding(image_bgr: np.ndarray, box: Sequence[float], padding_ratio: float = 0.10) -> np.ndarray:
	height, width = image_bgr.shape[:2]
	x1, y1, x2, y2 = [float(value) for value in box]
	box_width = x2 - x1
	box_height = y2 - y1

	pad_x = box_width * padding_ratio
	pad_y = box_height * padding_ratio

	left = max(int(x1 - pad_x), 0)
	top = max(int(y1 - pad_y), 0)
	right = min(int(x2 + pad_x), width)
	bottom = min(int(y2 + pad_y), height)

	if right <= left or bottom <= top:
		return image_bgr[max(int(y1), 0):min(int(y2), height), max(int(x1), 0):min(int(x2), width)]
	return image_bgr[top:bottom, left:right]


@torch.no_grad()
def classify_crop(model, crop_bgr: np.ndarray, transform, device: torch.device, class_names: Sequence[str]):
	crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
	crop_pil = Image.fromarray(crop_rgb)
	tensor = transform(crop_pil).unsqueeze(0).to(device)
	logits = model(tensor)
	probabilities = torch.softmax(logits, dim=1)[0]
	confidence, prediction = probabilities.max(dim=0)
	prediction_index = int(prediction.item())
	return {
		"class_name": class_names[prediction_index],
		"confidence": float(confidence.item()),
		"probabilities": {class_names[i]: float(probabilities[i].item()) for i in range(len(class_names))},
	}


def draw_text_box(image: np.ndarray, text: str, origin: Tuple[int, int], line_start: Tuple[int, int], color: Tuple[int, int, int]):
	x, y = origin
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.55
	thickness = 2
	(text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)

	padding = 6
	box_left = max(x, 0)
	box_top = max(y - text_height - 2 * padding, 0)
	box_right = min(box_left + text_width + 2 * padding, image.shape[1] - 1)
	box_bottom = min(box_top + text_height + 2 * padding + baseline, image.shape[0] - 1)

	cv2.rectangle(image, (box_left, box_top), (box_right, box_bottom), color, thickness=-1)
	cv2.line(image, line_start, (box_left, box_bottom), color, 2)
	cv2.putText(image, text, (box_left + padding, box_bottom - padding - baseline), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def annotate_yolo_only(result) -> np.ndarray:
	return result.plot()


def annotate_hybrid(image_bgr: np.ndarray, detections, cnn_model, cnn_transform, device: torch.device, class_names: Sequence[str]) -> np.ndarray:
	annotated = image_bgr.copy()

	if detections.boxes is None or len(detections.boxes) == 0:
		cv2.putText(annotated, "No faces detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
		return annotated

	for index, box in enumerate(detections.boxes):
		x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
		yolo_cls_id = int(box.cls.item())
		yolo_label = detections.names[yolo_cls_id]
		yolo_conf = float(box.conf.item()) * 100.0

		crop = crop_with_padding(image_bgr, (x1, y1, x2, y2), padding_ratio=0.12)
		if crop.size == 0:
			continue

		cnn_result = classify_crop(cnn_model, crop, cnn_transform, device, class_names)
		cnn_label = cnn_result["class_name"]
		cnn_conf = cnn_result["confidence"] * 100.0

		color = (0, 200, 0) if cnn_label == "with_mask" else (0, 0, 255)
		cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

		line_start = ((x1 + x2) // 2, y1)
		text_origin = (x2 + 8, max(y1 - 6, 32) + index * 28)
		text = f"YOLO {yolo_label} {yolo_conf:.1f}% | CNN {cnn_label} {cnn_conf:.1f}%"
		draw_text_box(annotated, text, text_origin, line_start, color)

	return annotated


def process_image(image_path: Path, yolo_weights: Path, cnn_weights: Path, save_dir: Path, imgsz: int, device_value: str | int, conf: float) -> None:
	YOLO = load_yolo_class()
	if not yolo_weights.exists():
		raise FileNotFoundError(f"YOLO weights file not found: {yolo_weights}")

	device = resolve_device(device_value)
	cnn_model, class_names, cnn_image_size = load_cnn_model(cnn_weights, device)
	cnn_transform = build_cnn_transform(cnn_image_size)

	yolo_model = YOLO(str(yolo_weights))
	image_bgr = load_image(image_path)
	detections = yolo_model.predict(source=str(image_path), imgsz=imgsz, conf=conf, device=device_value, verbose=False)[0]

	yolo_only = annotate_yolo_only(detections)
	hybrid = annotate_hybrid(image_bgr, detections, cnn_model, cnn_transform, device, class_names)

	save_dir.mkdir(parents=True, exist_ok=True)
	yolo_only_path = save_dir / f"{image_path.stem}_yolo_only.jpg"
	hybrid_path = save_dir / f"{image_path.stem}_hybrid.jpg"
	cv2.imwrite(str(yolo_only_path), yolo_only)
	cv2.imwrite(str(hybrid_path), hybrid)

	print(f"Saved YOLO-only image: {yolo_only_path}")
	print(f"Saved hybrid image: {hybrid_path}")

	print("\nDetections:")
	if detections.boxes is None or len(detections.boxes) == 0:
		print("  No faces detected.")
		return

	for box in detections.boxes:
		x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
		yolo_cls_id = int(box.cls.item())
		yolo_label = detections.names[yolo_cls_id]
		yolo_conf = float(box.conf.item()) * 100.0
		crop = crop_with_padding(image_bgr, (x1, y1, x2, y2), padding_ratio=0.12)
		cnn_result = classify_crop(cnn_model, crop, cnn_transform, device, class_names)
		print(f"  YOLO={yolo_label} {yolo_conf:.2f}% -> CNN={cnn_result['class_name']} {cnn_result['confidence'] * 100.0:.2f}%")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Run YOLO detection, crop faces, classify crops with CNN, and save separate outputs.")
	parser.add_argument("--image", type=Path, required=True, help="Input image path")
	parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS, help="Trained YOLO detector weights")
	parser.add_argument("--cnn-weights", type=Path, default=DEFAULT_CNN_WEIGHTS, help="Trained CNN classifier checkpoint")
	parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Directory to save annotated outputs")
	parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
	parser.add_argument("--device", default=0, help="YOLO/CNN device, e.g. 0 or cpu")
	parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if not args.image.exists():
		raise FileNotFoundError(f"Input image not found: {args.image}")

	process_image(
		image_path=args.image,
		yolo_weights=args.yolo_weights,
		cnn_weights=args.cnn_weights,
		save_dir=args.save_dir,
		imgsz=args.imgsz,
		device_value=args.device,
		conf=args.conf,
	)


if __name__ == "__main__":
	main()
