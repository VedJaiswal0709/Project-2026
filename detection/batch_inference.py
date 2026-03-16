import os
import cv2
import pandas as pd
from ultralytics import YOLO
import logging
import glob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_batch_inference(model_path, source_dirs, output_base_dir, conf_threshold=0.25):
    """
    Runs YOLOv8 detection on multiple directories mapping to 'Original', 'Weather Degraded', and 'Enhanced'.
    Saves bounding box images and a CSV report of detection confidences.
    
    Args:
        model_path (str): Path to YOLOv8 weights (best.pt).
        source_dirs (dict): Dictionary mapping category names to their folder paths.
        output_base_dir (str): Root folder to save results.
        conf_threshold (float): Minimum confidence for a detection.
    """
    logging.info(f"Loading YOLOv8 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logging.error(f"Failed to load model from {model_path}: {e}")
        return

    os.makedirs(output_base_dir, exist_ok=True)
    
    all_results_data = []

    for category, folder_path in source_dirs.items():
        if not os.path.exists(folder_path):
             logging.warning(f"Source folder '{folder_path}' for category '{category}' not found. Skipping.")
             continue
             
        # Create output directories for this category under results
        cat_output_dir = os.path.join(output_base_dir, category, 'images')
        os.makedirs(cat_output_dir, exist_ok=True)
        
        image_paths = glob.glob(os.path.join(folder_path, '*.*'))
        valid_images = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not valid_images:
            logging.info(f"No valid images in {folder_path}.")
            continue
            
        logging.info(f"Processing category '{category}' ({len(valid_images)} images)...")
        
        for img_path in valid_images:
            filename = os.path.basename(img_path)
            
            # Predict using YOLO
            results = model.predict(source=img_path, conf=conf_threshold, verbose=False)
            res = results[0]
            
            # Save bounding box image
            # The ultralytics result object's plot() method automatically draws boxes, labels, and confidences
            annotated_frame = res.plot()
            save_img_path = os.path.join(cat_output_dir, filename)
            cv2.imwrite(save_img_path, annotated_frame)
            
            # Extract confidence metrics
            boxes = res.boxes.cpu()
            num_detections = len(boxes)
            
            if num_detections > 0:
                # Average confidence across all bounding boxes in the image
                avg_confidence = float(boxes.conf.mean())
                max_confidence = float(boxes.conf.max())
                
                # Detailed tracking for each detection
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = res.names[cls_id]
                    indv_conf = float(box.conf[0])
                    
                    all_results_data.append({
                        'Category': category,
                        'Image_Name': filename,
                        'Class_Name': class_name,
                        'Confidence': indv_conf,
                        'Avg_Image_Confidence': avg_confidence,
                        'Max_Image_Confidence': max_confidence,
                        'Total_Detections': num_detections
                    })
            else:
                 # Record the image even if no detections were found
                 all_results_data.append({
                        'Category': category,
                        'Image_Name': filename,
                        'Class_Name': 'None',
                        'Confidence': 0.0,
                        'Avg_Image_Confidence': 0.0,
                        'Max_Image_Confidence': 0.0,
                        'Total_Detections': 0
                    })
                
    # Save the compiled CSV logic to the root results folder
    if all_results_data:
        df = pd.DataFrame(all_results_data)
        csv_path = os.path.join(output_base_dir, 'detection_confidence_report.csv')
        df.to_csv(csv_path, index=False)
        logging.info(f"Confidence summary report saved to {csv_path}")
        
    logging.info("Batch inference complete!")

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_WEIGHTS = "models/best.pt"
    OUTPUT_RESULTS_DIR = "results/batch_inference"
    
    # Define mapping of your conceptual categories to the folders where the images reside.
    # Adjust these paths as needed based on your project execution flow.
    INPUT_DIRECTORIES = {
        '1_Original': 'dataset/images/test',
        '2_Weather_Degraded_Rain': 'dataset/weather_simulations/rain_images',
        '2_Weather_Degraded_Fog': 'dataset/weather_simulations/fog_images',
        '3_Enhanced_CLAHE': 'enhancement_results/clahe',
        '3_Enhanced_Dehaze': 'enhancement_results/dehaze'
    }
    
    # Uncomment to run:
    # run_batch_inference(
    #     model_path=MODEL_WEIGHTS,
    #     source_dirs=INPUT_DIRECTORIES,
    #     output_base_dir=OUTPUT_RESULTS_DIR,
    #     conf_threshold=0.30
    # )
    
    logging.info("Inference pipeline generated. Setup 'INPUT_DIRECTORIES' inside main to execute.")
    pass
