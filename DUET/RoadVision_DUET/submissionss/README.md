# Submission Comparison

This folder contains a small utility for comparing two RoadVision DUET CSV files.

## Usage

```powershell
python compare_csvs.py yolo11m.csv roadvision_yolo26m_modified_all.csv
```

You can pass either:

- full paths to CSV files
- file names that live in `DUET/RoadVision_DUET/submissions`
- file names that live in this `submissionss` folder

The script prints:

- row counts
- image coverage overlap
- missing and extra `image_id` values
- per-class detection counts
- confidence statistics
- box statistics
- rows with empty or malformed prediction strings
