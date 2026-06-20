from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "faceMask" / "faceMask"
DATA_YAML = DATASET_DIR / "data.yaml"
RUNS_DIR = PROJECT_DIR / "runs"
SUPPORTED_SIZES = {"n", "s", "m", "l", "x"}


def load_yolo_class():
	try:
		return importlib.import_module("ultralytics").YOLO
	except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
		raise ModuleNotFoundError(
			"ultralytics is not installed. Install it first with: pip install ultralytics"
		) from exc


def resolve_model_weights(model_size: str, prefer_local_yolo26: bool = True) -> str:
	model_size = model_size.lower().strip()
	if model_size not in SUPPORTED_SIZES:
		raise ValueError(f"model_size must be one of {sorted(SUPPORTED_SIZES)}")

	local_weight = PROJECT_DIR / f"yolo26{model_size}.pt"
	if prefer_local_yolo26 and local_weight.exists():
		return str(local_weight)

	# Ultralytics can automatically download these official pretrained checkpoints.
	return f"yolo26{model_size}.pt"


def require_dataset() -> None:
	if not DATA_YAML.exists():
		raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")


def train(args: argparse.Namespace) -> Dict[str, Any]:
	require_dataset()
	YOLO = load_yolo_class()

	weights = resolve_model_weights(args.model_size, prefer_local_yolo26=not args.force_official)
	print(f"Dataset YAML: {DATA_YAML}")
	print(f"Model weights: {weights}")
	if weights.startswith(str(PROJECT_DIR)) and not Path(weights).exists():
		raise FileNotFoundError(
			f"Custom weights not found: {weights}. Put yolo26{args.model_size}.pt in {PROJECT_DIR} or use --force-official."
		)

	model = YOLO(weights)
	results = model.train(
		data=str(DATA_YAML),
		epochs=args.epochs,
		imgsz=args.imgsz,
		batch=args.batch,
		device=args.device,
		workers=args.workers,
		optimizer=args.optimizer,
		lr0=args.lr0,
		patience=args.patience,
		project=str(args.project),
		name=args.name,
		exist_ok=True,
		pretrained=True,
		verbose=True,
	)

	metrics = getattr(results, "results_dict", {})
	print(json.dumps(metrics, indent=2, default=str))
	return metrics


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
	require_dataset()
	YOLO = load_yolo_class()

	if not args.weights.exists():
		raise FileNotFoundError(f"Weights file not found: {args.weights}")

	model = YOLO(str(args.weights))
	metrics = model.val(data=str(DATA_YAML), split=args.split, imgsz=args.imgsz, device=args.device)
	summary = getattr(metrics, "results_dict", {})
	print(json.dumps(summary, indent=2, default=str))
	return summary


def predict(args: argparse.Namespace) -> None:
	YOLO = load_yolo_class()

	if not args.weights.exists():
		raise FileNotFoundError(f"Weights file not found: {args.weights}")

	model = YOLO(str(args.weights))
	for image_path in args.images:
		outputs = model.predict(
			source=str(image_path),
			imgsz=args.imgsz,
			conf=args.conf,
			device=args.device,
			verbose=False,
		)
		result = outputs[0]
		names = result.names

		print(f"\nImage: {image_path}")
		if result.boxes is None or len(result.boxes) == 0:
			print("  No faces detected.")
			continue

		for box in result.boxes:
			cls_id = int(box.cls.item())
			label = names[cls_id]
			confidence = float(box.conf.item()) * 100.0
			x1, y1, x2, y2 = box.xyxy[0].tolist()
			print(
				f"  {label}: {confidence:.2f}% | bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
			)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description="Train or evaluate a YOLO face-mask detector on the faceMask dataset."
	)
	parser.add_argument("--mode", choices=("train", "val", "predict"), default="train")
	parser.add_argument("--model-size", choices=sorted(SUPPORTED_SIZES), default="s")
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--imgsz", type=int, default=640)
	parser.add_argument("--batch", type=int, default=16)
	parser.add_argument("--device", default=0)
	parser.add_argument("--workers", type=int, default=2)
	parser.add_argument("--optimizer", default="AdamW")
	parser.add_argument("--lr0", type=float, default=0.001)
	parser.add_argument("--patience", type=int, default=20)
	parser.add_argument("--project", type=Path, default=RUNS_DIR)
	parser.add_argument("--name", default="face_mask_yolo")
	parser.add_argument("--weights", type=Path, default=PROJECT_DIR / "runs" / "detect" / "face_mask_yolo" / "weights" / "best.pt")
	parser.add_argument("--split", choices=("train", "val", "test"), default="test")
	parser.add_argument("--conf", type=float, default=0.25)
	parser.add_argument("--images", nargs="*", type=Path, default=[])
	parser.add_argument("--force-official", action="store_true", help="Use official Ultralytics weights like yolo11s.pt instead of local yolo26*.pt")
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.project = Path(args.project)
	args.weights = Path(args.weights)

	if args.mode == "train":
		train(args)
	elif args.mode == "val":
		evaluate(args)
	else:
		if not args.images:
			raise ValueError("Predict mode requires at least one image path passed via --images")
		predict(args)


if __name__ == "__main__":
	main()
