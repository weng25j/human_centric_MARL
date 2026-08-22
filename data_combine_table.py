import pandas as pd
import matplotlib.pyplot as plt

def generate_table_image(csv_path="combined_traffic_results.csv", output_path="comprehensive_results_table.png"):
    # Load your full dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Make sure it is in the same directory.")
        return

    # Filter for the 20% Penetration Rate to keep the table readable (include ALL-HDV at 0%)
    df = df[(df['Penetration Rate (%)'] == 20) | (df['Model'] == 'ALL-HDV')]
    
    # Select the most critical columns for the report
    cols_to_keep = [
        'Model', 'Demand (vph/lane)', 'Penetration Rate (%)', 
        'Equivalent Hourly Throughput (veh/h)', 'Mean Speed (m/s)', 
        'Avg Hard Braking', 'Mean Abs Jerk (m/s^3)', 
        'Duty Cycle (%)', 'Success Rate (%)', '90th-Percentile Delay (s)'
    ]
    
    # Filter columns safely
    available_cols = [c for c in cols_to_keep if c in df.columns]
    df = df[available_cols]

    # Clean up column headers for a cleaner visual
    rename_dict = {
        'Demand (vph/lane)': 'Demand\n(vph)',
        'Penetration Rate (%)': 'PR (%)',
        'Equivalent Hourly Throughput (veh/h)': 'Throughput\n(veh/h)',
        'Mean Speed (m/s)': 'Mean Speed\n(m/s)',
        'Avg Hard Braking': 'Hard\nBraking',
        'Mean Abs Jerk (m/s^3)': 'Mean Abs Jerk\n(m/s^3)',
        'Duty Cycle (%)': 'Duty Cycle\n(%)',
        'Success Rate (%)': 'Success\nRate (%)',
        '90th-Percentile Delay (s)': '90th-%ile\nDelay (s)'
    }
    df.rename(columns=rename_dict, inplace=True)
    
    # Sort for logical progression
    df.sort_values(by=['Demand\n(vph)', 'Model'], inplace=True)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.axis('off')
    ax.axis('tight')

    # Create the table
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)

    # Apply colors and formatting
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            # Header styling
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2f4f4f')
        else:
            # Highlight "TRAFFIC SHEPHERD" rows
            if "SHEPHERD" in str(df.iloc[row-1, 0]):
                cell.set_facecolor('#e6f2ff')
                cell.set_text_props(weight='bold')
            # Alternate row colors for readability
            elif row % 2 == 0:
                cell.set_facecolor('#f9f9f9')

    plt.title("Evaluation Metrics across Baselines (Mean ± 95% CI)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Successfully generated high-resolution table image: {output_path}")

if __name__ == "__main__":
    generate_table_image()