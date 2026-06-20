from ultralytics import YOLO
import torch
import cv2
import numpy as np
import sys
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

DEFAULT_IMAGE_PATH = BASE_DIR / "test.jpg"
SAMPLE_TEST_DIR = BASE_DIR / "faceMask" / "faceMask" / "SampleTest"

SAVE_DIR = BASE_DIR / "hybrid_results"


def resolve_image_path() -> Path:
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1]).expanduser()
        if candidate.exists():
            return candidate

        for base_dir in (BASE_DIR, PROJECT_ROOT):
            alternative = base_dir / candidate
            if alternative.exists():
                return alternative

    for image_dir in (SAMPLE_TEST_DIR, DEFAULT_IMAGE_PATH.parent):
        if image_dir.is_file() and image_dir.exists():
            return image_dir

        if image_dir.exists():
            for extension in ("*.jpg", "*.jpeg", "*.png"):
                for image_path in sorted(image_dir.glob(extension)):
                    return image_path

    for split_name in ("test", "val"):
        split_root = BASE_DIR / "cnn_dataset" / "cnn_dataset" / split_name
        for class_name in CLASS_NAMES:
            class_dir = split_root / class_name
            if not class_dir.exists():
                continue

            for extension in ("*.jpg", "*.jpeg", "*.png"):
                for image_path in sorted(class_dir.glob(extension)):
                    return image_path

    raise FileNotFoundError(
        "No input image found. Pass an image path as the first argument "
        "or place test.jpg in the YOLO directory."
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

# =====================================================
# LOAD YOLO
# =====================================================

print("Loading YOLO...")

if not YOLO_WEIGHTS.exists():
    raise FileNotFoundError(YOLO_WEIGHTS)

yolo_model = YOLO(str(YOLO_WEIGHTS))

# =====================================================
# LOAD EFFICIENTNET-B0
# =====================================================

print("Loading EfficientNet-B0...")

checkpoint = torch.load(
    str(CNN_WEIGHTS),
    map_location=DEVICE,
    weights_only=True
)

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

image_bgr = cv2.imread(str(IMAGE_PATH))

if image_bgr is None:
    raise FileNotFoundError(IMAGE_PATH)

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

# =====================================================
# YOLO DETECTION
# =====================================================

print("Running YOLO...")

results = yolo_model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    imgsz=640,
    verbose=False
)

result = results[0]

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
