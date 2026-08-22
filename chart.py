import csv
import matplotlib.pyplot as plt
import os

def generate_charts(csv_file="combined_traffic_results.csv", output_dir="comparison_charts"):
    # Create an output folder for the graphics
    os.makedirs(output_dir, exist_ok=True)

    # The exact column headers from your CSV that we want to plot
    metrics = [
        "Avg Throughput", "Mean Speed", "Mean TT", "Avg TTC", 
        "Avg Hard Brake", "Accel Var", "Mean Abs Jerk", "Wave Int", "Mean Delay"
    ]
    
    # NEW: Dictionary mapping each metric to its proper unit
    metric_units = {
        "Avg Throughput": "vehicles",
        "Mean Speed": "m/s",
        "Mean TT": "s",
        "Avg TTC": "violations/ep",
        "Avg Hard Brake": "events/ep",
        "Accel Var": "m/s²",
        "Mean Abs Jerk": "m/s³",
        "Wave Int": "speed variance",
        "Mean Delay": "s"
    }

    baselines = {}  # Dictionary to store ALL-HDV rows by Demand
    scenarios = {}  # Dictionary to store (Demand, PR) -> List of model rows

    print(f"Reading data from {csv_file}...")

    # 1. Parse the CSV Data (Using utf-8-sig to fix the BOM issue)
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip the empty separator rows
            if not row["Model"] or row["Model"].strip() == "":
                continue

            demand = row["Demand (vph)"]
            pr = row["Penetration Rate (%)"]
            model = row["Model"]

            # Anchor the baseline data
            if model == "ALL-HDV":
                baselines[demand] = row
            else:
                key = (demand, pr)
                if key not in scenarios:
                    scenarios[key] = []
                scenarios[key].append(row)

    # Standardized color palette for visual consistency across all charts
    model_colors = {
        "ALL-HDV": "#7f7f7f",          # Gray
        "RULE-CAV": "#1f77b4",         # Blue
        "SELFISH-CAV": "#aec7e8",      # Light Blue
        "FLAT-MARL": "#ff7f0e",        # Orange (DongChen baseline)
        "TRAFFIC SHEPHERD": "#2ca02c"  # Green
    }

    total_charts = 0

    # 2. Generate the Column Charts
    for (demand, pr), models_data in scenarios.items():
        # Prepend the correct ALL-HDV baseline to this specific group
        group_data = []
        if demand in baselines:
            group_data.append(baselines[demand])
        group_data.extend(models_data)

        model_names = [r["Model"] for r in group_data]
        colors = [model_colors.get(m, "#333333") for m in model_names]

        # Generate a chart for every single metric in this scenario
        for metric in metrics:
            values = [float(r[metric]) for r in group_data]

            plt.figure(figsize=(8, 5))
            bars = plt.bar(model_names, values, color=colors, edgecolor='black', linewidth=0.8)

            # NEW: Retrieve the unit and format the Y-axis label
            unit = metric_units.get(metric, "")
            plt.ylabel(f"{metric} ({unit})", fontsize=12)
            
            plt.title(f"{metric} Comparison ({demand} vph | {pr}% PR)", fontsize=13, fontweight='bold')
            plt.xticks(rotation=15, fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Add numerical value labels hovering slightly above each bar
            max_val = max(values) if values else 1
            for bar in bars:
                yval = bar.get_height()
                # Format to 1 decimal place for large numbers, 2 for small decimals
                label_text = f"{yval:.1f}" if yval > 10 else f"{yval:.2f}"
                plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max_val * 0.02),
                         label_text, ha='center', va='bottom', fontsize=10)

            plt.tight_layout()
            
            # Clean up the filename string and save
            safe_metric_name = metric.replace(" ", "_")
            filename = os.path.join(output_dir, f"chart_{demand}vph_{pr}PR_{safe_metric_name}.png")
            plt.savefig(filename, dpi=300)
            plt.close()
            
            total_charts += 1

    print(f"Success! Generated {total_charts} high-resolution charts in the '{output_dir}' folder.")

if __name__ == "__main__":
    generate_charts()