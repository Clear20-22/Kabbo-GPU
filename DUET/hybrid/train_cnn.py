from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
DEFAULT_DATA = PROJECT_DIR / "DUET" / "RoadVision_DUET" / "CNN_Crop_Classifier" / "dataset" / "crops"


def build_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {model_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RoadVision DUET CNN crop classifier.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default="convnext_tiny", choices=["convnext_tiny", "efficientnet_b3", "resnet50"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "runs" / "convnext_tiny")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tfms = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.18, contrast=0.20, saturation=0.18),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((args.imgsz, args.imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(args.data / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(args.data / "val", transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        scheduler.step()
        val_acc = correct / max(1, total)
        print(f"epoch={epoch:03d} train_loss={train_loss / len(train_ds):.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": train_ds.classes, "model_name": args.model}, args.output / "best.pt")

    print(f"Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()