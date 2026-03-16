import os
import pandas as pd
from ultralytics import YOLO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_f1_score(precision, recall):
    """
    Calculates the F1 Score given Precision and Recall.
    Handle division by zero safely.
    """
    # Precision and recall might be arrays for each class or single floats for mean averages
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_condition(model, condition_name, data_yaml_path):
    """
    Runs YOLO validation on a specific condition dataset and extracts metrics.
    """
    logging.info(f"--- Evaluating Condition: {condition_name} ({data_yaml_path}) ---")
    
    if not os.path.exists(data_yaml_path):
        logging.error(f"Dataset config {data_yaml_path} not found. Returning empty metrics.")
        return None
        
    try:
        # Run validation. verbose=False prevents it from overflowing console output
        metrics = model.val(data=data_yaml_path, verbose=False)
        
        # Extract mean metrics across all classes
        mean_precision = float(metrics.box.mp)
        mean_recall = float(metrics.box.mr)
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        
        # Calculate F1 Score based on mean precision and recall
        f1_score = calculate_f1_score(mean_precision, mean_recall)
        
        # Compile dictionary
        return {
            'Condition': condition_name,
            'Precision': round(mean_precision, 4),
            'Recall': round(mean_recall, 4),
            'F1_Score': round(f1_score, 4),
            'mAP50': round(map50, 4),
            'mAP50-95': round(map50_95, 4)
        }
        
    except Exception as e:
        logging.error(f"Error evaluating condition {condition_name}: {e}")
        return None

def run_performance_comparison(model_path, data_configs, output_csv_path):
    """
    Evaluates the YOLOv8 model across multiple datasets (Original, Degraded, Enhanced).
    Generates a comparison table and exports it as a CSV.
    
    Args:
        model_path (str): Path to trained YOLOv8 weights (e.g. models/best.pt).
        data_configs (dict): Mapping of Dataset Condition Name to its dataset.yaml path.
        output_csv_path (str): Where to save the resulting CSV.
    """
    logging.info(f"Loading YOLOv8 model from {model_path} for evaluation...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        return

    all_metrics = []

    for condition_name, yaml_path in data_configs.items():
        metrics_dict = evaluate_condition(model, condition_name, yaml_path)
        if metrics_dict:
            all_metrics.append(metrics_dict)
            logging.info(f"Results for {condition_name}: mAP50={metrics_dict['mAP50']:.4f}, F1={metrics_dict['F1_Score']:.4f}")

    if all_metrics:
        # Generate DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
        logging.info(f"\nEvaluation complete. Comparison table exported to: {output_csv_path}")
        
        # Print comparison table in console for quick review
        print("\n--- PERFORMANCE COMPARISON TABLE ---")
        print(df.to_string(index=False))
        return df
    else:
        logging.warning("No metrics were successfully computed.")
        return None

if __name__ == "__main__":
    # --- Configuration ---
    MODEL_WEIGHTS = "models/best.pt"
    OUTPUT_CSV = "results/dataset_performance_comparison.csv"
    
    # Define mapping of datasets to their YOLO yaml configurations.
    # To run this properly, you need `dataset.yaml` files generated for each scenario.
    DATASET_CONFIGS = {
        '1_Original_Clear': 'dataset/original.yaml',
        '2_Degraded_Rain': 'dataset/weather_simulations/rain.yaml',
        '2_Degraded_Fog': 'dataset/weather_simulations/fog.yaml',
        '3_Enhanced_CLAHE': 'enhancement_results/clahe.yaml',
        '3_Enhanced_Dehaze': 'enhancement_results/dehaze.yaml'
    }
    
    # Uncomment to execute:
    # df = run_performance_comparison(MODEL_WEIGHTS, DATASET_CONFIGS, OUTPUT_CSV)
    
    logging.info("Evaluation module ready. Make sure YAML configs are present and uncomment execution block.")
    pass
