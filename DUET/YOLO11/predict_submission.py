from __future__ import annotations

import argparse
import csv
import gc
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent
ROADVISION_DIR = PROJECT_DIR / "DUET" / "RoadVision_DUET"
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "yolo11m_800" / "weights" / "best.pt"


def format_float(value: float) -> str:
    """Formats float coordinates up to 6 decimal places cleanly without trailing zeros."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def predict_single_image(model: YOLO, image_path: str, args: argparse.Namespace):
    """Runs prediction on a single image with an aggressive OOM CPU-fallback."""
    predict_kwargs = dict(
        source=image_path,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )

    if str(args.device).lower() != "cpu":
        predict_kwargs["device"] = args.device
        predict_kwargs["half"] = True  # FP16 optimization saves VRAM

    try:
        # Return the first (and only) result from the single-image prediction
        return model.predict(**predict_kwargs)[0]
    
    except torch.OutOfMemoryError:
        # The true OOM catch. Clear everything out.
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()

        print(f"\n⚠️ OOM on {Path(image_path).name}. Running CPU fallback for this frame...")
        
        predict_kwargs["device"] = "cpu"
        predict_kwargs["half"] = False
        return model.predict(**predict_kwargs)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a RoadVision DUET Kaggle-style submission from YOLO weights.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--sample", type=Path, default=ROADVISION_DIR / "test" / "sample_submission.csv")
    parser.add_argument("--test-images", type=Path, default=ROADVISION_DIR / "test" / "images")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "submission.csv")
    
    # --- STRATEGIC ARGUMENT ADJUSTMENTS ---
    # Fixed imgsz to 1248 to avoid dynamic padding overhead (1248 is a multiple of 32)
    parser.add_argument("--imgsz", type=int, default=1248)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--device", default=0)
    parser.add_argument("--max-det", type=int, default=1000)
    
    # --- PHYSICAL FILTER SETTING ---
    parser.add_argument("--min-dim", type=float, default=0.001)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}.")
    if not args.test_images.exists():
        raise FileNotFoundError(f"Test image folder not found: {args.test_images}")

    print(f"📦 Loading fine-tuned model: {args.weights.name}")
    model = YOLO(str(args.weights))
    
    test_files = sorted(
        [p for p in args.test_images.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name,
    )
    
    total_frames = len(test_files)
    print(f"🚀 Running sequential inference against {total_frames} frames...")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    total_written_detections = 0
    skipped_ghost_boxes = 0

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "PredictionString"])
        writer.writeheader()
        
        # We loop through files ONE by ONE to maintain strict VRAM control
        for idx, image_path in enumerate(test_files, 1):
            image_id = image_path.stem
            
            # Print progress cleanly every 25 images
            if idx % 25 == 0 or idx == total_frames:
                print(f"   Processing: {idx}/{total_frames}...")

            # Run inference explicitly wrapped in torch.inference_mode() for max efficiency
            with torch.inference_mode():
                result = predict_single_image(model, str(image_path), args)
            
            parts: list[str] = []
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                cls_values = boxes.cls.detach().cpu().tolist()
                conf_values = boxes.conf.detach().cpu().tolist()
                xywhn_values = boxes.xywhn.detach().cpu().tolist()
                
                # Sort indices: highest confidence scores first
                order = sorted(range(len(conf_values)), key=lambda i: conf_values[i], reverse=True)
                
                for i in order:
                    w = max(0.0, min(1.0, float(xywhn_values[i][2])))
                    h = max(0.0, min(1.0, float(xywhn_values[i][3])))
                    
                    # 🚫 CRITICAL DEFENSIVE SANITY CHECK: Purge zero-area boxes
                    if w <= args.min_dim or h <= args.min_dim:
                        skipped_ghost_boxes += 1
                        continue
                        
                    cls = int(cls_values[i])
                    conf = max(0.0, min(1.0, float(conf_values[i])))
                    x = max(0.0, min(1.0, float(xywhn_values[i][0])))
                    y = max(0.0, min(1.0, float(xywhn_values[i][1])))
                    
                    parts.extend([
                        str(cls),
                        format_float(conf),
                        format_float(x),
                        format_float(y),
                        format_float(w),
                        format_float(h)
                    ])
                    total_written_detections += 1
                    
            writer.writerow({"image_id": image_id, "PredictionString": " ".join(parts)})
            
            # Explicitly delete the result and box tensors to free VRAM for the next loop
            del result
            del boxes

    print("\n" + "="*50)
    print("🏁 INFERENCE RUN COMPLETE")
    print("="*50)
    print(f"📁 Saved to: {args.output}")
    print(f"📊 Valid Bounding Boxes Serialized: {total_written_detections}")
    if skipped_ghost_boxes > 0:
        print(f"⚠️  Microscopic / Zero-Area Boxes Blocked: {skipped_ghost_boxes}")
    print("="*50)


if __name__ == "__main__":
    main()