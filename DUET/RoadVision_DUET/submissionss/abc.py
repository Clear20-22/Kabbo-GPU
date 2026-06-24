import csv
from pathlib import Path

# Set your paths here
INPUT_CSV = Path(r"C:\Users\User\Sojib\Kabbo-GPU\DUET\RoadVision_DUET\submissionss\yolo11l.csv")
OUTPUT_CSV = Path(r"C:\Users\User\Sojib\Kabbo-GPU\DUET\RoadVision_DUET\submissionss\yolo11l_cleaned.csv")

MIN_DIMENSION = 0.001 # Drop boxes with width or height below this

def clean_prediction_string(pred_str):
    if not pred_str.strip():
        return ""
    
    parts = pred_str.strip().split()
    valid_parts = []
    
    for i in range(0, len(parts), 6):
        cls = parts[i]
        conf = parts[i+1]
        x = parts[i+2]
        y = parts[i+3]
        w = float(parts[i+4])
        h = float(parts[i+5])
        
        # Check if the box has actual physical structure
        if w > MIN_DIMENSION and h > MIN_DIMENSION:
            valid_parts.extend([cls, conf, x, y, f"{w:.6f}", f"{h:.6f}"])
            
    return " ".join(valid_parts)

print(f"🧹 Scrubbing zero-area boxes from: {INPUT_CSV.name}...")
removed_boxes = 0

with open(INPUT_CSV, "r", newline="", encoding="utf-8") as infile, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as outfile:
    
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    
    for row in reader:
        orig_pred = row["PredictionString"]
        cleaned_pred = clean_prediction_string(orig_pred)
        
        # Count difference
        orig_count = len(orig_pred.strip().split()) // 6
        clean_count = len(cleaned_pred.strip().split()) // 6
        removed_boxes += (orig_count - clean_count)
        
        row["PredictionString"] = cleaned_pred
        writer.writerow(row)

print(f"✨ Successfully generated: {OUTPUT_CSV.name}")
print(f"🚫 Dropped {removed_boxes} corrupt bounding boxes total.")