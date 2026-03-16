import os
import csv
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(message)s')

def evaluate_model(model_path='models/best.pt', data_yaml='dataset/dataset.yaml', output_csv='results/evaluation_metrics.csv'):
    print("\n--- Starting Model Evaluation ---")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return
        
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset yaml not found at {data_yaml}")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Evaluating on test dataset defined in: {data_yaml}")
    
    # Run validation step on test split (split='test' in ultralytics)
    try:
        metrics = model.val(data=data_yaml, split='test', plots=True)
        
        # Extract metrics
        precision = metrics.box.p.mean()
        recall = metrics.box.r.mean()
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        # Calculate F1 Score (harmonic mean of precision and recall)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)
        
        print("\n--- Evaluation Results ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1_score:.4f}")
        print(f"mAP@50:    {map50:.4f}")
        print(f"mAP@50-95: {map50_95:.4f}")
        print("--------------------------\n")
        
        # Save to CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Score'])
            writer.writerow(['Precision', f"{precision:.4f}"])
            writer.writerow(['Recall', f"{recall:.4f}"])
            writer.writerow(['F1 Score', f"{f1_score:.4f}"])
            writer.writerow(['mAP50', f"{map50:.4f}"])
            writer.writerow(['mAP50-95', f"{map50_95:.4f}"])
            
        print(f"Metrics saved to {output_csv}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_model()
