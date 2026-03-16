import os
import sys
import time
import logging

# Append project root to path so we can import internal modules easily
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the modular functions created across the project
from dataset.pipeline import split_dataset, extract_and_process_frames
from dataset.annotations import generate_dataset_yaml, verify_dataset_integrity
from weather_simulation.simulate_weather import process_directory as simulate_weather
from enhancement.enhance_image import process_enhancements
from training.train_yolo import train_yolov8_model, export_and_finalize_model
from detection.batch_inference import run_batch_inference
from evaluation.evaluate_metrics import run_performance_comparison
from results.data_visualization import generate_all_visualizations

# ---------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------

# Create a custom logger for the master pipeline
logger = logging.getLogger('MasterPipeline')
logger.setLevel(logging.INFO)

# Create console handler with beautiful formatting
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('\n[%(asctime)s] === %(levelname)s ===\n%(message)s\n', datefmt='%H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)

def print_stage_header(stage_number, stage_name):
    """Prints a highly visible header to track pipeline progress."""
    border = "=" * 60
    logger.info(f"{border}\n>>> STAGE {stage_number}: {stage_name.upper()} <<<\n{border}")

# ---------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------

def run_full_pipeline():
    logger.info("Initializing Indian Traffic Vehicle Detection Under Adverse Weather Pipeline...")
    start_time = time.time()
    
    # ---------------------------------------------------------
    # System Paths Configuration
    # (Adjust these paths based on where your raw data sits)
    # ---------------------------------------------------------
    RAW_VIDEO = "dataset/raw/traffic_cam.mp4"
    TEMP_FRAMES_DIR = "dataset/temp_frames"
    DATASET_ROOT = "dataset"
    DATA_YAML = "dataset/dataset.yaml"
    
    WEATHER_OUTPUT = "dataset/weather_simulations"
    ENHANCE_OUTPUT = "enhancement_results"
    
    MODELS_DIR = "models"
    FINAL_WEIGHTS = os.path.join(MODELS_DIR, "best.pt")
    
    BATCH_INFERENCE_DIR = "results/batch_inference"
    METRICS_CSV = "results/dataset_performance_comparison.csv"
    GRAPHS_DIR = "results/graphs"
    
    try:
        # ---------------------------------------------------------
        # STAGE 1: Dataset Preparation & Preprocessing
        # ---------------------------------------------------------
        print_stage_header(1, "Dataset Preparation")
        
        # 1a. Extract Frames from video
        # (Uncomment next line once you have `traffic_cam.mp4` placed in `dataset/raw/`)
        # extract_and_process_frames(RAW_VIDEO, TEMP_FRAMES_DIR, frame_skip=15)
        
        # 1b. Split into YOLO Train/Val/Test
        # (Uncomment next line after extraction and annotating your data)
        # split_dataset(TEMP_FRAMES_DIR, DATASET_ROOT)
        
        # 1c. Generate YAML and Verify Annotations
        # generate_dataset_yaml(DATA_YAML, DATASET_ROOT)
        # verify_dataset_integrity(DATASET_ROOT)
        
        logger.info("Dataset Preparation stage completed (currently in placeholder mode via comments).")


        # ---------------------------------------------------------
        # STAGE 2: Weather Simulation
        # ---------------------------------------------------------
        print_stage_header(2, "Weather Simulation")
        
        # We simulate weather over the training images to train the model to be robust
        clear_train_dir = os.path.join(DATASET_ROOT, 'images', 'train')
        if os.path.exists(clear_train_dir):
            simulate_weather(clear_train_dir, WEATHER_OUTPUT)
        else:
             logger.warning(f"Skipping Weather sim: {clear_train_dir} not found.")


        # ---------------------------------------------------------
        # STAGE 3: Image Enhancement
        # ---------------------------------------------------------
        print_stage_header(3, "Image Enhancement")
        
        # Example: Let's enhance the generated Fog images
        fog_dir = os.path.join(WEATHER_OUTPUT, 'fog_images')
        if os.path.exists(fog_dir):
            process_enhancements(fog_dir, ENHANCE_OUTPUT)
        else:
            logger.warning(f"Skipping Enhancement: {fog_dir} not found.")


        # ---------------------------------------------------------
        # STAGE 4: YOLOv8 Training
        # ---------------------------------------------------------
        print_stage_header(4, "YOLOv8 Model Training")
        
        # (Uncomment below to run 50-epoch training on the created dataset.yaml)
        # best_weights = train_yolov8_model(
        #     data_yaml_path=DATA_YAML,
        #     epochs=50, batch_size=16, img_size=640, optimizer='AdamW'
        # )
        # if best_weights:
        #     export_and_finalize_model(best_weights, MODELS_DIR)
        
        logger.info("Training script execution logged (currently disabled via comments).")


        # ---------------------------------------------------------
        # STAGE 5: Vehicle Detection (Batch Inference)
        # ---------------------------------------------------------
        print_stage_header(5, "Batch Vehicle Detection")
        
        # Define the dataset subsets we want to run our model against
        inference_dirs = {
            'Original': os.path.join(DATASET_ROOT, 'images', 'test'),
            'Degraded_Rain': os.path.join(WEATHER_OUTPUT, 'rain_images'),
            'Enhanced_CLAHE': os.path.join(ENHANCE_OUTPUT, 'clahe')
        }
        
        if os.path.exists(FINAL_WEIGHTS):
            run_batch_inference(FINAL_WEIGHTS, inference_dirs, BATCH_INFERENCE_DIR)
        else:
             logger.warning(f"Skipping Inference: Trained weights not found at {FINAL_WEIGHTS}")


        # ---------------------------------------------------------
        # STAGE 6: Performance Evaluation
        # ---------------------------------------------------------
        print_stage_header(6, "Performance Evaluation")
        
        # Map conditions to their respective config files generated earlier
        eval_configs = {
            'Baseline': DATA_YAML,
            # 'Rain_Degraded': 'dataset/weather_simulations/rain.yaml', # Paths to generated YAMLS
            # 'CLAHE_Enhanced': 'enhancement_results/clahe.yaml'
        }
        
        if os.path.exists(FINAL_WEIGHTS):
            # run_performance_comparison(FINAL_WEIGHTS, eval_configs, METRICS_CSV)
            pass
        else:
            logger.warning("Skipping Evaluation: Weights not found.")


        # ---------------------------------------------------------
        # STAGE 7: Data Visualization
        # ---------------------------------------------------------
        print_stage_header(7, "Graph Visualization")
        
        if os.path.exists(METRICS_CSV):
             generate_all_visualizations(METRICS_CSV, GRAPHS_DIR)
        else:
             logger.warning(f"Skipping Visualization: Metric CSV not found at {METRICS_CSV}")

        
        # --- End of Pipeline ---
        end_time = time.time()
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        
        logger.info("=" * 60)
        logger.info(f"MASTER PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        logger.info(f"Total processing time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        logger.info("=" * 60)
        
    except Exception as e:
         logger.error(f"PIPELINE FAILED: An unexpected error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    # Ensure our working directory is the root of the project before starting
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run_full_pipeline()
    pass
