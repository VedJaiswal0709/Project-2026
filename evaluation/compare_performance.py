import os
import yaml
import csv
from ultralytics import YOLO

def create_temp_yaml(base_yaml_path, new_test_path, temp_yaml_path):
    with open(base_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    data['test'] = new_test_path
    
    with open(temp_yaml_path, 'w') as f:
        yaml.dump(data, f)

def evaluate_and_record(model, data_yaml_path, condition_name, writer):
    print(f"\n--- Evaluating Condition: {condition_name} ---")
    try:
        metrics = model.val(data=data_yaml_path, split='test', verbose=False)
        mAP50 = metrics.box.map50
        precision = metrics.box.p.mean()
        recall = metrics.box.r.mean()
        
        writer.writerow([condition_name, precision, recall, mAP50])
        print(f"Results -> mAP50: {mAP50:.4f} | P: {precision:.4f} | R: {recall:.4f}")
    except Exception as e:
        print(f"Failed to evaluate {condition_name}: {e}")

def compare_performance(model_path='models/best.pt', base_yaml='dataset/dataset.yaml', output_csv='results/comparison_results.csv'):
    print("\n=== Comprehensive Performance Comparison ===")
    
    if not os.path.exists(model_path):
        print("Model not found. Train the model first.")
        return
        
    model = YOLO(model_path)
    temp_yaml = 'dataset/temp_eval.yaml'
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Condition', 'Precision', 'Recall', 'mAP50'])
        
        # 1. Evaluate Baseline (Original Test Set)
        evaluate_and_record(model, base_yaml, 'Baseline (Original)', writer)
        
        # 2. Evaluate Weather Degraded
        conditions = ['rain', 'fog', 'snow', 'wind']
        for cond in conditions:
            test_path = f"images/test_weather/{cond}"
            if os.path.exists(os.path.join('dataset', test_path)):
                create_temp_yaml(base_yaml, test_path, temp_yaml)
                evaluate_and_record(model, temp_yaml, f"Weather: {cond}", writer)
                
        # 3. Evaluate Enhanced
        techniques = ['hist_eq', 'clahe', 'gamma', 'dehaze', 'sharpen']
        for cond in conditions:
            for tech in techniques:
                test_path = f"images/test_enhanced/{cond}/{tech}"
                if os.path.exists(os.path.join('dataset', test_path)):
                    create_temp_yaml(base_yaml, test_path, temp_yaml)
                    evaluate_and_record(model, temp_yaml, f"Enhanced: {cond} + {tech}", writer)
                    
    # Cleanup temp yaml
    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
        
    print(f"\nComparison completed. Results saved to {output_csv}")

if __name__ == "__main__":
    compare_performance()
