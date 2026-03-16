import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_style():
    """Applies a clean, modern aesthetic to all matplotlib/seaborn charts."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 150
    })

def plot_accuracy_bar_chart(csv_path, output_dir):
    """
    Generates a Bar Chart comparing mAP50 detection accuracy across 
    Original, Weather Degraded, and Enhanced conditions.
    """
    if not os.path.exists(csv_path):
        logging.warning("Metrics CSV not found. Skipping Bar Chart.")
        return

    df = pd.read_csv(csv_path)
    if 'mAP50' not in df.columns or 'Condition' not in df.columns:
        return

    plt.figure()
    
    # Custom color mapping based on expected condition prefixes
    colors = []
    for cond in df['Condition']:
        if 'Original' in cond: colors.append('#2E86AB')     # Blue
        elif 'Degraded' in cond: colors.append('#D36060')   # Red
        elif 'Enhanced' in cond: colors.append('#568259')   # Green
        else: colors.append('#A9A9A9')                      # Gray
        
    bars = plt.bar(df['Condition'], df['mAP50'], color=colors, edgecolor='black', linewidth=1)
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

    plt.title('Vehicle Detection Accuracy (mAP50) vs Conditions')
    plt.ylabel('mAP@0.5')
    plt.xlabel('Dataset Condition')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'accuracy_comparison_barchart.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Accuracy Bar Chart saved to: {save_path}")

def plot_weather_impact_line_chart(csv_path, output_dir):
    """
    Generates a Line Chart tracking the impact of different weather 
    severities or types on the F1 Score.
    """
    if not os.path.exists(csv_path):
         logging.warning("Metrics CSV not found. Skipping Line Chart.")
         return

    df = pd.read_csv(csv_path)
    if 'F1_Score' not in df.columns:
         return
         
    # Example logic: order metrics theoretically from Clear -> Degraded -> Enhanced
    # To plot a continuous line, we sort them conceptually or alphabetically
    df_sorted = df.sort_values(by='Condition')

    plt.figure()
    plt.plot(df_sorted['Condition'], df_sorted['F1_Score'], marker='o', linestyle='-', color='#C73E1D', markersize=8, linewidth=2.5)
    
    plt.title('Weather Impact on Tracking Performance (F1 Score)')
    plt.ylabel('F1 Score')
    plt.xlabel('Dataset State')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'weather_impact_linechart.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Weather Impact Line Chart saved to: {save_path}")

def plot_detection_heatmap(output_dir):
    """
    Generates a simulated Heatmap showing average detection confidence 
    across different vehicle classes and weather conditions.
    (This function uses simulated data meant to be replaced by merged CSV pivot tables)
    """
    plt.figure(figsize=(10, 8))
    
    # Example Target Classes and Conditions for the matrix
    classes = ['Car', 'Bus', 'Truck', 'Motorcycle', 'Auto Rickshaw']
    conditions = ['Clear', 'Rain', 'Fog', 'Enhanced (CLAHE)']
    
    # Simulated confidence metric data (0.0 to 1.0)
    data = np.array([
        [0.95, 0.92, 0.96, 0.88, 0.90], # Clear
        [0.75, 0.78, 0.82, 0.60, 0.70], # Rain
        [0.65, 0.68, 0.72, 0.50, 0.60], # Fog
        [0.85, 0.87, 0.90, 0.78, 0.82]  # Enhanced
    ])
    
    # Create the heatmap using Seaborn
    sns.heatmap(data, annot=True, fmt=".2f", cmap="YlGnBu", 
                xticklabels=classes, yticklabels=conditions, 
                cbar_kws={'label': 'Detection Confidence'})
                
    plt.title('Vehicle Detection Confidence Heatmap')
    plt.xlabel('Vehicle Class')
    plt.ylabel('Weather / Processing Condition')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'detection_heatmap.png')
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Detection Heatmap saved to: {save_path}")

def generate_all_visualizations(metrics_csv, graphs_dir):
    """
    Master function to run all visualizations sequentially and save them.
    """
    os.makedirs(graphs_dir, exist_ok=True)
    set_style()
    
    plot_accuracy_bar_chart(metrics_csv, graphs_dir)
    plot_weather_impact_line_chart(metrics_csv, graphs_dir)
    plot_detection_heatmap(graphs_dir)
    
    logging.info("All visual charts successfully generated!")

if __name__ == "__main__":
    # --- Configuration ---
    # Path to the CSV generated by evaluation_metrics.py
    METRICS_CSV_FILE = "results/dataset_performance_comparison.csv"
    
    # Directory to store the output graphs
    OUTPUT_GRAPHS_DIR = "results/graphs"
    
    # Uncomment to execute:
    # generate_all_visualizations(METRICS_CSV_FILE, OUTPUT_GRAPHS_DIR)
    
    logging.info("Visualization module ready. Ensure CSV exists and uncomment execution script.")
    pass
