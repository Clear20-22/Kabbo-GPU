import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

# --- CONFIGURATION ---
CSV_PATH = Path(__file__).resolve().parent / "submission.csv"

CLASS_NAMES = [
    "Rickshaw", "Motorcycle", "Tempu", "Sedan Car", "Pickup",
    "Microbus", "Mini Bus", "Mini Truck", "Agro Use",
    "Medium Truck", "Large Bus", "Heavy Truck", "Trailer"
]

def analyze_submission(file_path: Path):
    if not file_path.exists():
        print(f"❌ Error: Could not find the file at {file_path}")
        return

    total_images = 0
    empty_images = 0
    total_detections = 0
    
    class_counts = Counter()
    class_confidences = defaultdict(list)
    
    widths, heights, areas = [], [], []

    with file_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total_images += 1
            pred_string = (row.get("PredictionString") or "").strip()
            
            if not pred_string:
                empty_images += 1
                continue
                
            parts = pred_string.split()
            # Each detection block must have exactly 6 elements: cls, conf, x, y, w, h
            if len(parts) % 6 != 0:
                print(f"⚠️ Warning: Malformed prediction string row found for image: {row.get('image_id')}")
                continue

            for i in range(0, len(parts), 6):
                try:
                    cls_id = int(float(parts[i]))
                    conf = float(parts[i+1])
                    w = float(parts[i+4])
                    h = float(parts[i+5])
                    
                    total_detections += 1
                    class_counts[cls_id] += 1
                    class_confidences[cls_id].append(conf)
                    
                    widths.append(w)
                    heights.append(h)
                    areas.append(w * h)
                except (ValueError, IndexError):
                    continue

    # --- PRINT STATISTICAL REPORT ---
    print("\n" + "="*50)
    print(f"📊 STATISTICAL PROFILE FOR: {file_path.name}")
    print("="*50)
    print(f"🖼️  Total Images Checked:      {total_images}")
    print(f"🫙  Images with No Objects:   {empty_images}")
    print(f"🎯 Total Objects Detected:    {total_detections}")
    if total_images > 0:
        print(f"📈 Avg Objects Per Image:     {total_detections / total_images:.2f}")
    print("-" * 50)
    
    if total_detections == 0:
        print("❌ No detections found in this submission file.")
        return

    print("🗂️  DETECTIONS PER CLASS:")
    print(f"  {'Class ID':<9} {'Class Name':<15} | {'Count':<8} | {'Avg Confidence':<14}")
    print("  " + "-"*50)
    
    # Sort by Class ID order
    for cls_id in range(len(CLASS_NAMES)):
        count = class_counts.get(cls_id, 0)
        confs = class_confidences.get(cls_id, [])
        avg_conf = mean(confs) if confs else 0.0
        print(f"  [{cls_id:02d}]     {CLASS_NAMES[cls_id]:<15} | {count:<8} | {avg_conf:.4f}")
        
    print("-" * 50)
    print("📐 BOX GEOMETRY METRICS:")
    print(f"  • Widths:  Min={min(widths):.4f}, Max={max(widths):.4f}, Mean={mean(widths):.4f}")
    print(f"  • Heights: Min={min(heights):.4f}, Max={max(heights):.4f}, Mean={mean(heights):.4f}")
    print(f"  • Areas:   Min={min(areas):.6f}, Max={max(areas):.4f}, Mean={mean(areas):.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # You can change this filename string if you want to inspect a different output file
    analyze_submission(CSV_PATH)