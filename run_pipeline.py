import os
import subprocess
import time

def run_script(script_path, desc):
    print(f"\n==============================================")
    print(f"==> EXECUTING: {desc}")
    print(f"==> SCRIPT: {script_path}")
    print(f"==============================================\n")
    
    start_time = time.time()
    try:
        # Run script assuming it's correctly mapped in PYTHONPATH or run from project root
        result = subprocess.run(["python", script_path], check=True)
        print(f"\n[INFO] {desc} completed successfully in {time.time() - start_time:.2f} seconds.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {desc} failed with exit code {e.returncode}.")
        exit(1)

def run_pipeline():
    print("Starting Main Execution Pipeline for Indian Traffic Vehicle Detection")
    print("Ensuring you are running this from the `project` root directory...")
    
    scripts = [
        ("dataset/validate_dataset.py", "Dataset Validation"),
        ("training/train_yolo.py", "YOLO Model Training"),
        ("evaluation/evaluate_model.py", "Baseline Model Evaluation"),
        ("weather_simulation/simulate_weather.py", "Weather Robustness Simulation"),
        ("enhancement/enhance_images.py", "Image Enhancement Simulation"),
        ("evaluation/compare_performance.py", "Performance Comparison"),
        ("results/generate_graphs.py", "Results Graph Generation")
    ]
    
    for script_path, desc in scripts:
        if os.path.exists(script_path):
            run_script(script_path, desc)
        else:
            print(f"[ERROR] Script not found: {script_path}")
            exit(1)
            
    print("\n==============================================")
    print("==> ALL TASKS COMPLETED SUCCESSFULLY")
    print("==> Visualizations available in `results/graphs/`")
    print("==> Weights available in `models/`")
    print("==============================================")

if __name__ == "__main__":
    run_pipeline()
