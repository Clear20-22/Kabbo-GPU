from ultralytics import YOLO
import torch
import cv2
import numpy as np
import sys
import time
import traceback
from pathlib import Path
from PIL import Image

from torchvision import transforms
from torchvision.models import efficientnet_b0

# =====================================================
# PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

YOLO_WEIGHTS = BASE_DIR / "runs" / "face_mask_yolo" / "weights" / "best.pt"
CNN_WEIGHTS = BASE_DIR / "efficientnetb0_mask_status.pth"

DEFAULT_IMAGE_PATH = (
    BASE_DIR
    / "faceMask"
    / "facemask"
    / "test"
    / "images"
    / "maksssksksss106_png.rf.d412b83c46a92de2daaa6d4b57426160.jpg"
)

SAVE_DIR = BASE_DIR / "hybrid_results"


def log(message: str) -> None:
    print(f"[DEBUG] {message}")


def resolve_image_path() -> Path:
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            return candidate

        for base_dir in (BASE_DIR, PROJECT_ROOT):
            alternative = base_dir / candidate
            if alternative.exists():
                return alternative

    if DEFAULT_IMAGE_PATH.exists():
        return DEFAULT_IMAGE_PATH

    raise FileNotFoundError(
        "No input image found. Pass an image path as the first argument "
        "or make sure the default test image exists."
    )

# =====================================================
# CLASSES
# =====================================================

CLASS_NAMES = [
    "mask_weared_incorrect",
    "with_mask",
    "without_mask"
]

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

log(f"Torch version: {torch.__version__}")
log(f"CUDA available: {torch.cuda.is_available()}")
log(f"Selected device: {DEVICE}")
log(f"Ultralytics version: {YOLO.__module__.split('.')[0] if hasattr(YOLO, '__module__') else 'unknown'}")

# =====================================================
# LOAD YOLO
# =====================================================

print("Loading YOLO...")
log(f"YOLO weight path: {YOLO_WEIGHTS}")
log(f"YOLO weight exists: {YOLO_WEIGHTS.exists()}")

if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(YOLO_WEIGHTS)

start_time = time.perf_counter()
try:
    log("Starting YOLO model load")
    yolo_model = YOLO(str(YOLO_WEIGHTS))
    log("Finished YOLO model load")
except Exception:
    log("YOLO model load failed")
    traceback.print_exc()
    raise
finally:
    elapsed = time.perf_counter() - start_time
    print(f"YOLO load time: {elapsed:.3f}s")

# =====================================================
# LOAD EFFICIENTNET-B0
# =====================================================

print("Loading EfficientNet-B0...")

log(f"CNN weight path: {CNN_WEIGHTS}")
log(f"CNN weight exists: {CNN_WEIGHTS.exists()}")

start_time = time.perf_counter()
try:
    log("Starting EfficientNet checkpoint load")
    checkpoint = torch.load(
        str(CNN_WEIGHTS),
        map_location=DEVICE,
        weights_only=True
    )
    log("Finished EfficientNet checkpoint load")
except Exception:
    log("EfficientNet checkpoint load failed")
    traceback.print_exc()
    raise
finally:
    elapsed = time.perf_counter() - start_time
    print(f"EfficientNet load time: {elapsed:.3f}s")

checkpoint_class_names = checkpoint.get("class_names", CLASS_NAMES)

cnn_model = efficientnet_b0(weights=None)

in_features = cnn_model.classifier[1].in_features

cnn_model.classifier[1] = torch.nn.Sequential(
    torch.nn.Dropout(p=0.35),
    torch.nn.Linear(
        in_features,
        len(checkpoint_class_names)
    )
)

cnn_model.load_state_dict(
    checkpoint["model_state"]
)

cnn_model.to(DEVICE)
cnn_model.eval()

# =====================================================
# TRANSFORM
# =====================================================

IMAGE_SIZE = checkpoint.get("image_size", 224)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =====================================================
# LOAD IMAGE
# =====================================================

IMAGE_PATH = resolve_image_path()
log(f"Resolved image path: {IMAGE_PATH}")
log(f"Image exists: {IMAGE_PATH.exists()}")

start_time = time.perf_counter()
try:
    log("Starting OpenCV image read")
    image_bgr = cv2.imread(str(IMAGE_PATH))
    log("Finished OpenCV image read")
except Exception:
    log("OpenCV image read failed")
    traceback.print_exc()
    raise
finally:
    elapsed = time.perf_counter() - start_time
    print(f"Image read time: {elapsed:.3f}s")

if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

log(f"Image shape (BGR): {image_bgr.shape}")

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

# =====================================================
# YOLO DETECTION
# =====================================================

print("Running YOLO...")

start_time = time.perf_counter()
try:
    log("Starting YOLO prediction")
    results = yolo_model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        imgsz=640,
        verbose=True
    )
    log("Finished YOLO prediction")
except Exception:
    log("YOLO prediction failed")
    traceback.print_exc()
    raise
finally:
    elapsed = time.perf_counter() - start_time
    print(f"YOLO prediction time: {elapsed:.3f}s")

result = results[0]
log(f"YOLO boxes available: {result.boxes is not None}")
if result.boxes is not None:
    log(f"YOLO box count: {len(result.boxes)}")

# =====================================================
# YOLO IMAGE
# =====================================================

yolo_output = result.plot()

SAVE_DIR.mkdir(
    exist_ok=True,
    parents=True
)

YOLO_OUTPUT_PATH = SAVE_DIR / "yolo_result.jpg"

cv2.imwrite(
    str(YOLO_OUTPUT_PATH),
    yolo_output
)

# =====================================================
# HYBRID IMAGE
# =====================================================

hybrid_output = image_bgr.copy()

# =====================================================
# DETECTIONS
# =====================================================

if result.boxes is not None:

    for box in result.boxes:

        x1,y1,x2,y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        # ----------------------------------
        # Crop Face
        # ----------------------------------

        face_crop = image_rgb[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # ----------------------------------
        # CNN Prediction
        # ----------------------------------

        face_pil = Image.fromarray(face_crop)

        tensor = transform(face_pil)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(DEVICE)

        start_time = time.perf_counter()
        try:
            log(f"Starting CNN inference for box {(x1, y1, x2, y2)}")
            with torch.no_grad():

                logits = cnn_model(tensor)

                probs = torch.softmax(
                    logits,
                    dim=1
                )

                conf, pred = torch.max(
                    probs,
                    dim=1
                )
            log("Finished CNN inference")
        except Exception:
            log("CNN inference failed")
            traceback.print_exc()
            raise
        finally:
            elapsed = time.perf_counter() - start_time
            print(f"CNN prediction time: {elapsed:.3f}s")

        pred_class = checkpoint_class_names[
            pred.item()
        ]

        confidence = conf.item() * 100

        # ----------------------------------
        # Color
        # ----------------------------------

        if pred_class == "with_mask":
            color = (0,255,0)

        elif pred_class == "without_mask":
            color = (0,0,255)

        else:
            color = (0,255,255)

        # ----------------------------------
        # Draw
        # ----------------------------------

        cv2.rectangle(
            hybrid_output,
            (x1,y1),
            (x2,y2),
            color,
            2
        )

        label = (
            f"{pred_class} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            hybrid_output,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        print(
            f"{pred_class} "
            f"{confidence:.2f}%"
        )

# =====================================================
# SAVE
# =====================================================

HYBRID_OUTPUT_PATH = (
    SAVE_DIR / "hybrid_result.jpg"
)

cv2.imwrite(
    str(HYBRID_OUTPUT_PATH),
    hybrid_output
)

# =====================================================
# SHOW
# =====================================================

cv2.imshow(
    "YOLO Detection",
    yolo_output
)

cv2.imshow(
    "Hybrid Prediction",
    hybrid_output
)

cv2.waitKey(0)
cv2.destroyAllWindows()

print()
print("===================================")
print("YOLO Result Saved:")
print(YOLO_OUTPUT_PATH)

print()

print("Hybrid Result Saved:")
print(HYBRID_OUTPUT_PATH)
print("===================================")
