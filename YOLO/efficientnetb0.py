from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


try:
	LANCZOS = Image.Resampling.LANCZOS
	BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # pragma: no cover - older Pillow fallback
	LANCZOS = Image.LANCZOS
	BILINEAR = Image.BILINEAR


PROJECT_DIR = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_DIR / "cnn_dataset" / "cnn_dataset"
DEFAULT_CHECKPOINT = PROJECT_DIR / "efficientnetb0_mask_status.pth"
DEFAULT_HISTORY = PROJECT_DIR / "efficientnetb0_history.csv"
CLASS_NAMES = ["mask_weared_incorrect", "with_mask", "without_mask"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = False
	torch.backends.cudnn.benchmark = True


@dataclass
class DimensionSummary:
	count: int
	min_size: Tuple[int, int]
	median_size: Tuple[int, int]
	max_size: Tuple[int, int]
	unique_sizes: int


class LetterboxResize:
	def __init__(self, size: int, fill: Tuple[int, int, int] = (114, 114, 114)) -> None:
		self.size = size
		self.fill = fill

	def __call__(self, image: Image.Image) -> Image.Image:
		rgb_image = image.convert("RGB")
		return ImageOps.pad(
			rgb_image,
			(self.size, self.size),
			method=LANCZOS,
			color=self.fill,
			centering=(0.5, 0.5),
		)


def analyze_split_dimensions(split_dir: Path) -> DimensionSummary:
	sizes: List[Tuple[int, int]] = []
	for image_path in split_dir.rglob("*.jpg"):
		with Image.open(image_path) as image:
			sizes.append(image.size)

	if not sizes:
		raise FileNotFoundError(f"No JPG images found under {split_dir}")

	widths = [width for width, _ in sizes]
	heights = [height for _, height in sizes]
	return DimensionSummary(
		count=len(sizes),
		min_size=(min(widths), min(heights)),
		median_size=(int(median(widths)), int(median(heights))),
		max_size=(max(widths), max(heights)),
		unique_sizes=len(set(sizes)),
	)


def inspect_dataset_dimensions(data_root: Path) -> Dict[str, Dict[str, DimensionSummary]]:
	report: Dict[str, Dict[str, DimensionSummary]] = {}
	for split_name in ("train", "val", "test"):
		split_dir = data_root / split_name
		if not split_dir.exists():
			continue

		report[split_name] = {}
		for class_name in sorted(path.name for path in split_dir.iterdir() if path.is_dir()):
			class_dir = split_dir / class_name
			report[split_name][class_name] = analyze_split_dimensions(class_dir)

	return report


def print_dimension_report(report: Dict[str, Dict[str, DimensionSummary]]) -> None:
	print("Dataset image-size summary")
	print("-" * 80)
	for split_name, class_report in report.items():
		print(split_name.upper())
		for class_name, summary in class_report.items():
			print(
				f"  {class_name:24s} count={summary.count:4d} "
				f"min={summary.min_size} median={summary.median_size} "
				f"max={summary.max_size} unique={summary.unique_sizes}"
			)
		print()


def recommend_image_size(report: Dict[str, Dict[str, DimensionSummary]]) -> int:
	medians: List[int] = []
	max_sides: List[int] = []
	for class_report in report.values():
		for summary in class_report.values():
			medians.append(max(summary.median_size))
			max_sides.append(max(summary.max_size))

	if not medians:
		return 320

	median_side = int(median(medians))
	largest_side = max(max_sides)

	if median_side <= 48 or largest_side <= 192:
		return 320
	if median_side <= 96:
		return 384
	return 224


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
	train_transform = transforms.Compose(
		[
			LetterboxResize(image_size),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.RandomRotation(degrees=7, interpolation=BILINEAR, fill=(114, 114, 114)),
			transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05, hue=0.02),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)

	eval_transform = transforms.Compose(
		[
			LetterboxResize(image_size),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)
	return train_transform, eval_transform


def build_datasets(data_root: Path, image_size: int):
	train_transform, eval_transform = build_transforms(image_size)
	train_dataset = datasets.ImageFolder(data_root / "train", transform=train_transform)
	val_dataset = datasets.ImageFolder(data_root / "val", transform=eval_transform)
	test_dataset = datasets.ImageFolder(data_root / "test", transform=eval_transform)
	return train_dataset, val_dataset, test_dataset


def build_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
	targets = [label for _, label in dataset.samples]
	class_counts = Counter(targets)
	sample_weights = [1.0 / class_counts[label] for label in targets]
	return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_model(num_classes: int) -> nn.Module:
	weights = EfficientNet_B0_Weights.DEFAULT
	model = efficientnet_b0(weights=weights)
	in_features = model.classifier[1].in_features
	model.classifier[1] = nn.Sequential(
		nn.Dropout(p=0.35),
		nn.Linear(in_features, num_classes),
	)
	return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
	trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
	total = sum(parameter.numel() for parameter in model.parameters())
	return trainable, total


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
	model.eval()
	total_loss = 0.0
	correct = 0
	total = 0
	num_classes = model.classifier[1][-1].out_features if isinstance(model.classifier[1], nn.Sequential) else model.classifier[1].out_features
	confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, targets)

		total_loss += loss.item() * targets.size(0)
		predictions = logits.argmax(dim=1)
		correct += (predictions == targets).sum().item()
		total += targets.size(0)

		for target, prediction in zip(targets.view(-1), predictions.view(-1)):
			confusion[target.long(), prediction.long()] += 1

	accuracy = correct / max(total, 1)
	macro_precision, macro_recall, macro_f1 = macro_metrics(confusion)
	return {
		"loss": total_loss / max(total, 1),
		"accuracy": accuracy,
		"macro_precision": macro_precision,
		"macro_recall": macro_recall,
		"macro_f1": macro_f1,
		"confusion_matrix": confusion,
	}


def macro_metrics(confusion: torch.Tensor) -> Tuple[float, float, float]:
	precision_values = []
	recall_values = []
	f1_values = []

	for class_index in range(confusion.size(0)):
		true_positive = confusion[class_index, class_index].item()
		false_positive = confusion[:, class_index].sum().item() - true_positive
		false_negative = confusion[class_index, :].sum().item() - true_positive

		precision = true_positive / max(true_positive + false_positive, 1)
		recall = true_positive / max(true_positive + false_negative, 1)
		f1 = (2.0 * precision * recall) / max(precision + recall, 1e-12)

		precision_values.append(precision)
		recall_values.append(recall)
		f1_values.append(f1)

	return (
		sum(precision_values) / len(precision_values),
		sum(recall_values) / len(recall_values),
		sum(f1_values) / len(f1_values),
	)


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler,
	device: torch.device,
) -> Dict[str, float]:
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0

	for images, targets in loader:
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)

		optimizer.zero_grad(set_to_none=True)
		with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
			logits = model(images)
			loss = criterion(logits, targets)

		scaler.scale(loss).backward()
		scaler.step(optimizer)
		scaler.update()

		running_loss += loss.item() * targets.size(0)
		predictions = logits.argmax(dim=1)
		correct += (predictions == targets).sum().item()
		total += targets.size(0)

	return {
		"loss": running_loss / max(total, 1),
		"accuracy": correct / max(total, 1),
	}


def save_checkpoint(
	path: Path,
	model: nn.Module,
	class_names: Sequence[str],
	image_size: int,
	epoch: int,
	best_metrics: Dict[str, float],
) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	torch.save(
		{
			"model_state": model.state_dict(),
			"class_names": list(class_names),
			"image_size": image_size,
			"epoch": epoch,
			"metrics": best_metrics,
			"architecture": "efficientnet_b0",
		},
		path,
	)


def load_checkpoint(path: Path, device: torch.device):
	checkpoint = torch.load(path, map_location=device)
	class_names = checkpoint.get("class_names", CLASS_NAMES)
	image_size = int(checkpoint.get("image_size", 320))
	model = build_model(len(class_names)).to(device)
	model.load_state_dict(checkpoint["model_state"])
	model.eval()
	return model, class_names, image_size, checkpoint


@torch.no_grad()
def predict_image(model: nn.Module, image_path: Path, image_size: int, device: torch.device, class_names: Sequence[str]):
	transform = transforms.Compose(
		[
			LetterboxResize(image_size),
			transforms.ToTensor(),
			transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
		]
	)
	with Image.open(image_path) as image:
		tensor = transform(image).unsqueeze(0).to(device)
	logits = model(tensor)
	probabilities = torch.softmax(logits, dim=1)[0]
	confidence, prediction = probabilities.max(dim=0)
	return {
		"path": str(image_path),
		"class_index": int(prediction.item()),
		"class_name": class_names[int(prediction.item())],
		"confidence": float(confidence.item()),
		"probabilities": {
			class_names[index]: float(probabilities[index].item()) for index in range(len(class_names))
		},
	}


def write_history(history_path: Path, rows: List[Dict[str, float]]) -> None:
	history_path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = sorted({key for row in rows for key in row.keys()})
	with history_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def resolve_device(force_cpu: bool = False) -> torch.device:
	if force_cpu:
		return torch.device("cpu")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def train(args: argparse.Namespace) -> None:
	if not DATA_ROOT.exists():
		raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

	seed_everything(args.seed)

	dimension_report = inspect_dataset_dimensions(DATA_ROOT)
	print_dimension_report(dimension_report)

	image_size = args.image_size or recommend_image_size(dimension_report)
	print(f"Selected CNN input size: {image_size}x{image_size}")

	train_dataset, val_dataset, test_dataset = build_datasets(DATA_ROOT, image_size)
	class_names = [name for name, _ in sorted(train_dataset.class_to_idx.items(), key=lambda item: item[1])]

	train_loader = DataLoader(
		train_dataset,
		batch_size=args.batch_size,
		sampler=build_sampler(train_dataset),
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)
	test_loader = DataLoader(
		test_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers,
		pin_memory=torch.cuda.is_available(),
		persistent_workers=args.num_workers > 0,
	)

	device = resolve_device(args.cpu)
	model = build_model(len(class_names)).to(device)
	trainable, total = count_parameters(model)
	print(f"Model parameters: trainable={trainable:,} total={total:,}")

	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.4,
		patience=2,
		threshold=1e-4,
	)
	scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

	best_score = -math.inf
	best_loss = math.inf
	patience_counter = 0
	history_rows: List[Dict[str, float]] = []

	for epoch in range(1, args.epochs + 1):
		train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
		val_metrics = evaluate(model, val_loader, criterion, device)
		scheduler.step(val_metrics["loss"])

		row = {
			"epoch": float(epoch),
			"train_loss": float(train_metrics["loss"]),
			"train_accuracy": float(train_metrics["accuracy"]),
			"val_loss": float(val_metrics["loss"]),
			"val_accuracy": float(val_metrics["accuracy"]),
			"val_macro_precision": float(val_metrics["macro_precision"]),
			"val_macro_recall": float(val_metrics["macro_recall"]),
			"val_macro_f1": float(val_metrics["macro_f1"]),
		}
		history_rows.append(row)

		print(
			f"Epoch {epoch:03d} | "
			f"train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.4f} | "
			f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
			f"f1 {val_metrics['macro_f1']:.4f}"
		)

		improved = val_metrics["accuracy"] > best_score or (
			math.isclose(val_metrics["accuracy"], best_score) and val_metrics["loss"] < best_loss
		)
		if improved:
			best_score = val_metrics["accuracy"]
			best_loss = val_metrics["loss"]
			patience_counter = 0
			save_checkpoint(
				args.checkpoint,
				model,
				class_names,
				image_size,
				epoch,
				{
					"val_accuracy": float(val_metrics["accuracy"]),
					"val_loss": float(val_metrics["loss"]),
					"val_macro_f1": float(val_metrics["macro_f1"]),
				},
			)
		else:
			patience_counter += 1

		if patience_counter >= args.patience:
			print(f"Early stopping triggered at epoch {epoch}.")
			break

	write_history(args.history, history_rows)

	if args.evaluate_test:
		checkpoint_model, checkpoint_class_names, checkpoint_image_size, _ = load_checkpoint(args.checkpoint, device)
		test_metrics = evaluate(checkpoint_model, test_loader, criterion, device)
		print("Test metrics")
		print(json.dumps({k: float(v) if isinstance(v, (float, int)) else v for k, v in test_metrics.items() if k != "confusion_matrix"}, indent=2))
		print("Class names:", checkpoint_class_names)
		print("Image size:", checkpoint_image_size)


def predict(args: argparse.Namespace) -> None:
	device = resolve_device(args.cpu)
	model, class_names, image_size, checkpoint = load_checkpoint(args.checkpoint, device)
	print(json.dumps({"checkpoint_metrics": checkpoint.get("metrics", {}), "image_size": image_size, "classes": class_names}, indent=2))

	image_paths = [Path(path) for path in args.images]
	for image_path in image_paths:
		result = predict_image(model, image_path, image_size, device, class_names)
		print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train or run an EfficientNet-B0 CNN for YOLO crops.")
	parser.add_argument("--mode", choices=("train", "predict"), default="train")
	parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
	parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
	parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
	parser.add_argument("--image-size", type=int, default=0)
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--label-smoothing", type=float, default=0.05)
	parser.add_argument("--patience", type=int, default=5)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--num-workers", type=int, default=2)
	parser.add_argument("--cpu", action="store_true")
	parser.add_argument("--evaluate-test", action="store_true")
	parser.add_argument("--images", nargs="*", default=[])
	return parser


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	args.checkpoint = Path(args.checkpoint)
	args.history = Path(args.history)

	if args.mode == "train":
		train(args)
		return

	if not args.images:
		raise ValueError("Predict mode requires at least one image path in --images")
	predict(args)


if __name__ == "__main__":
	main()
