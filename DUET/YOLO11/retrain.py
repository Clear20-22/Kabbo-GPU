from ultralytics import YOLO

def main():
    # 1. Load the model weights you want to fine-tune
    # Pointing to your specific runs directory
    model = YOLO(r"C:\Users\User\Sojib\Kabbo-GPU\DUET\YOLO11\runs\yolo11m_800\weights\best.pt")

    # 2. Run the fine-tuning training
    model.train(
        data=r"C:\Users\User\Sojib\Kabbo-GPU\DUET\RoadVision_DUET\sample\heavy_traffic_dataset\data.yaml",
        epochs=5,              # Adjust based on convergence
        imgsz=960,             # Keep consistent with your high-res needs
        batch=4,                # Adjust based on your GPU's VRAM
        lr0=0.0005,              # Lower learning rate for fine-tuning
        optimizer="AdamW",
        device=0,
        cache="disk",
        name="yolo11m_heavy_traffic_finetune"
    )

if __name__ == "__main__":
    main()