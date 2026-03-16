import os
import shutil
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_yolov8(
    data_yaml_path='dataset/dataset.yaml',
    model_version='yolov8s.pt',
    epochs=50,
    batch_size=16,
    img_size=640,
    optimizer='AdamW',
    lr0=0.001,
    project_dir='runs/train',
    name='indian_traffic_model',
    export_dir='models'
):
    logging.info("Starting YOLOv8 Training Pipeline...")
    logging.info(f"Using dataset: {data_yaml_path}")
    
    if not os.path.exists(data_yaml_path):
        logging.error(f"Dataset config {data_yaml_path} not found!")
        return

    # Load pretrained model
    logging.info(f"Loading pretrained weights: {model_version}")
    model = YOLO(model_version)
    
    # Train the model
    logging.info("Starting training...")
    try:
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            optimizer=optimizer,
            lr0=lr0,
            project=project_dir,
            name=name,
            device='', # Auto device selection
            plots=True, # Generates graphs
            save=True   # Save weights
        )
        
        logging.info("Training completed.")
        
        # Determine paths
        run_dir = os.path.join(project_dir, name)
        best_model_path = os.path.join(run_dir, 'weights', 'best.pt')
        
        if os.path.exists(best_model_path):
            os.makedirs(export_dir, exist_ok=True)
            export_path = os.path.join(export_dir, 'best.pt')
            shutil.copy(best_model_path, export_path)
            logging.info(f"Best model saved to {export_path}")
        else:
            logging.error(f"Could not find best.pt at {best_model_path}")
            
    except Exception as e:
        logging.error(f"Training failed: {e}")

if __name__ == "__main__":
    # Assumes run from project root
    train_yolov8()
