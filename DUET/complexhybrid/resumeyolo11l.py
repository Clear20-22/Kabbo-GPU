# resume_training.py
from ultralytics import YOLO

def resume_run():
    # 1. Point directly to the last saved checkpoint before the crash
    LAST_CHECKPOINT = r"C:\Users\User\Sojib\Kabbo-GPU\DUET\complexhybrid\runs\yolo11l_modified_all-2\weights\last.pt"
    
    # 2. Load the exact state of the model at epoch 55
    model = YOLO(LAST_CHECKPOINT)
    
    # 3. Resume the pipeline safely
    print("🔄 Resuming YOLO11l training from Epoch 55...")
    model.train(resume=True)

if __name__ == "__main__":
    resume_run()