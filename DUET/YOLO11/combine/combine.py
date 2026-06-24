import os
import argparse
import logging

import pandas as pd
import numpy as np

from tqdm import tqdm

from ensemble_boxes import (
    weighted_boxes_fusion,
    nms,
    soft_nms
)

# ----------------------------------------------------
# Logging
# ----------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s"
)

# ----------------------------------------------------
# Parse PredictionString
# ----------------------------------------------------

def parse_prediction_string(pred_str):

    boxes = []
    scores = []
    labels = []

    if pd.isna(pred_str) or pred_str == "":
        return boxes, scores, labels

    parts = pred_str.strip().split()

    if len(parts) % 6 != 0:
        return boxes, scores, labels

    for i in range(0, len(parts), 6):

        try:
            cls = int(float(parts[i]))
            conf = float(parts[i + 1])

            xc = float(parts[i + 2])
            yc = float(parts[i + 3])

            w = float(parts[i + 4])
            h = float(parts[i + 5])

            if w <= 0 or h <= 0:
                continue

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
            labels.append(cls)

        except Exception:
            continue

    return boxes, scores, labels


# ----------------------------------------------------
# Convert Back
# ----------------------------------------------------

def boxes_to_prediction_string(boxes, scores, labels):

    result = []

    for box, score, label in zip(boxes, scores, labels):

        x1, y1, x2, y2 = box

        w = x2 - x1
        h = y2 - y1

        xc = x1 + w / 2
        yc = y1 + h / 2

        result.extend([
            str(int(label)),
            f"{score:.6f}",
            f"{xc:.2f}",
            f"{yc:.2f}",
            f"{w:.2f}",
            f"{h:.2f}"
        ])

    return " ".join(result)


# ----------------------------------------------------
# Normalize
# ----------------------------------------------------

def normalize_boxes(boxes):

    if len(boxes) == 0:
        return []

    arr = np.array(boxes)

    max_x = max(arr[:, 2].max(), 1)
    max_y = max(arr[:, 3].max(), 1)

    norm = []

    for b in boxes:

        x1, y1, x2, y2 = b

        norm.append([
            x1 / max_x,
            y1 / max_y,
            x2 / max_x,
            y2 / max_y
        ])

    return norm


def denormalize_boxes(boxes, max_x, max_y):

    out = []

    for b in boxes:

        out.append([
            b[0] * max_x,
            b[1] * max_y,
            b[2] * max_x,
            b[3] * max_y
        ])

    return out


# ----------------------------------------------------
# Analysis
# ----------------------------------------------------

def analyze(df, name):

    total = 0

    class_count = {}
    class_conf = {}

    for pred in df["PredictionString"]:

        _, scores, labels = parse_prediction_string(pred)

        total += len(labels)

        for l, s in zip(labels, scores):

            class_count[l] = class_count.get(l, 0) + 1

            if l not in class_conf:
                class_conf[l] = []

            class_conf[l].append(s)

    logging.info(f"{name}")
    logging.info(f"Total detections: {total}")

    for c in sorted(class_count):

        avg_conf = np.mean(class_conf[c])

        logging.info(
            f"Class {c}: "
            f"{class_count[c]} detections, "
            f"avg conf={avg_conf:.4f}"
        )


# ----------------------------------------------------
# Ensemble
# ----------------------------------------------------

def ensemble_image(
        boxes1,
        scores1,
        labels1,
        boxes2,
        scores2,
        labels2,
        method="wbf",
        iou_thr=0.55,
        skip_box_thr=0.001):

    all_boxes = boxes1 + boxes2

    if len(all_boxes) == 0:
        return [], [], []

    arr = np.array(all_boxes)

    max_x = max(arr[:, 2].max(), 1)
    max_y = max(arr[:, 3].max(), 1)

    b1 = normalize_boxes(boxes1)
    b2 = normalize_boxes(boxes2)

    boxes_list = [b1, b2]
    scores_list = [scores1, scores2]
    labels_list = [labels1, labels2]

    weights = [2.0, 1.0]

    if method == "wbf":

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr
        )

    elif method == "nms":

        boxes, scores, labels = nms(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=iou_thr
        )

    else:

        boxes, scores, labels = soft_nms(
            boxes_list,
            scores_list,
            labels_list,
            iou_thr=iou_thr
        )

    boxes = denormalize_boxes(boxes, max_x, max_y)

    return boxes, scores, labels


# ----------------------------------------------------
# Main
# ----------------------------------------------------

def run_ensemble(
        csv1,
        csv2,
        output_csv,
        conf_thr,
        method,
        iou_thr,
        skip_box_thr):

    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    analyze(df1, "YOLO11m")
    analyze(df2, "YOLO26")

    image_ids = df1["image_id"].tolist()

    results = []

    for _, row1 in tqdm(
            df1.iterrows(),
            total=len(df1),
            desc="Ensembling"):

        image_id = row1["image_id"]

        row2 = df2[df2.image_id == image_id]

        if len(row2) == 0:
            continue

        row2 = row2.iloc[0]

        b1, s1, l1 = parse_prediction_string(
            row1["PredictionString"]
        )

        b2, s2, l2 = parse_prediction_string(
            row2["PredictionString"]
        )

        idx1 = [i for i, s in enumerate(s1) if s >= conf_thr]
        idx2 = [i for i, s in enumerate(s2) if s >= conf_thr]

        b1 = [b1[i] for i in idx1]
        s1 = [s1[i] for i in idx1]
        l1 = [l1[i] for i in idx1]

        b2 = [b2[i] for i in idx2]
        s2 = [s2[i] for i in idx2]
        l2 = [l2[i] for i in idx2]

        boxes, scores, labels = ensemble_image(
            b1,
            s1,
            l1,
            b2,
            s2,
            l2,
            method,
            iou_thr,
            skip_box_thr
        )

        pred_str = boxes_to_prediction_string(
            boxes,
            scores,
            labels
        )

        results.append({
            "image_id": image_id,
            "PredictionString": pred_str
        })

    pd.DataFrame(results).to_csv(
        output_csv,
        index=False
    )

    logging.info(
        f"Saved -> {output_csv}"
    )


# ----------------------------------------------------
# Threshold Sweep
# ----------------------------------------------------

def threshold_search(args):

    thresholds = [
        0.01,
        0.03,
        0.05,
        0.07,
        0.10,
        0.15,
        0.20
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for thr in thresholds:

        out_file = os.path.join(
            args.output_dir,
            f"{args.method}_{thr:.2f}.csv"
        )

        logging.info(
            f"Running threshold={thr}"
        )

        run_ensemble(
            args.csv1,
            args.csv2,
            out_file,
            thr,
            args.method,
            args.iou,
            args.skip
        )


# ----------------------------------------------------
# CLI
# ----------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv1",
        required=True
    )

    parser.add_argument(
        "--csv2",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="outputs"
    )

    parser.add_argument(
        "--method",
        default="wbf",
        choices=[
            "wbf",
            "nms",
            "soft_nms"
        ]
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.55
    )

    parser.add_argument(
        "--skip",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    threshold_search(args)


if __name__ == "__main__":
    main()