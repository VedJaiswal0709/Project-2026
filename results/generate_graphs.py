import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_graphs(csv_path='results/comparison_results.csv', output_dir='results/graphs'):
    print("\n--- Generating Visualization Graphs ---")
    if not os.path.exists(csv_path):
        print(f"Error: Results CSV {csv_path} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Extract baseline
    baseline = df[df['Condition'] == 'Baseline (Original)'].iloc[0]
    
    # 1. Weather Impact Analysis
    weather_df = df[df['Condition'].str.startswith('Weather:')]
    if not weather_df.empty:
        plt.figure(figsize=(10, 6))
        
        conditions = ['Baseline'] + [c.replace('Weather: ', '') for c in weather_df['Condition']]
        mAPs = [baseline['mAP50']] + list(weather_df['mAP50'])
        
        plt.bar(conditions, mAPs, color=['blue'] + ['red']*len(weather_df))
        plt.title('Impact of Weather Conditions on Detection mAP50')
        plt.ylabel('mAP@50')
        plt.ylim(0, 1.0)
        
        for i, v in enumerate(mAPs):
            plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'weather_impact.png'))
        plt.close()
        print("Generated weather_impact.png")
        
    # 2. Enhancement Improvement Analysis
    conditions_list = ['rain', 'fog', 'snow', 'wind']
    for cond in conditions_list:
        enh_df = df[df['Condition'].str.startswith(f'Enhanced: {cond}')]
        weather_row = df[df['Condition'] == f'Weather: {cond}']
        
        if not enh_df.empty and not weather_row.empty:
            plt.figure(figsize=(12, 6))
            
            labels = ['Degraded'] + [c.split('+ ')[1] for c in enh_df['Condition']]
            mAPs = [weather_row.iloc[0]['mAP50']] + list(enh_df['mAP50'])
            
            plt.bar(labels, mAPs, color=['red'] + ['green']*len(enh_df))
            plt.title(f'Enhancement Techniques Improvement on {cond.upper()} condition')
            plt.ylabel('mAP@50')
            plt.ylim(0, 1.0)
            
            for i, v in enumerate(mAPs):
                plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'enhancement_{cond}.png'))
            plt.close()
            print(f"Generated enhancement_{cond}.png")

    print(f"All graphs generated in {output_dir}")

if __name__ == "__main__":
    generate_graphs()
