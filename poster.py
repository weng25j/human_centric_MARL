import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_mean_std(val_str):
    """Safely split 'Mean ± StdDev' strings into float tuples."""
    if pd.isna(val_str):
        return 0.0, 0.0
    if isinstance(val_str, str) and '±' in val_str:
        parts = val_str.split('±')
        return float(parts[0].strip()), float(parts[1].strip())
    try:
        return float(val_str), 0.0
    except ValueError:
        return 0.0, 0.0

def load_and_clean_data(csv_path="combined_traffic_results.csv"):
    """Loads the data directly from the specified CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the data file: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Identify metric columns (those containing ±)
    metric_cols = [col for col in df.columns if df[col].dtype == object and df[col].str.contains('±', na=False).any()]
    
    # Split metrics into _mean and _std columns
    for col in metric_cols:
        df[col + '_mean'] = df[col].apply(lambda x: parse_mean_std(x)[0])
        df[col + '_std'] = df[col].apply(lambda x: parse_mean_std(x)[1])
        
    # Standardize model names to uppercase to avoid case-sensitivity issues
    df['Model'] = df['Model'].str.strip().str.upper()
        
    return df, metric_cols

# ==========================================
# 1. GENERATE COMPREHENSIVE BAR CHARTS FOR ALL METRICS
# ==========================================
def plot_comparative_bar_chart(df, metric_col, title_prefix, ylabel, filename):
    pr_labels = ['5% PR', '10% PR', '20% PR']
    target_prs = [5, 10, 20]
    x = np.arange(len(pr_labels))
    width = 0.2  # Shrunk width to fit 4 bars per PR group
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    
    # Configuration for the 4 bar-plotted models: (csv_name, display_name, color, offset)
    model_configs = [
        ('RULE-CAV', 'RULE-CAV', '#d62728', -1.5),                # Red
        ('SELFISH-CAV', 'SELFISH-CAV', '#9467bd', -0.5),          # Purple
        ('DONGCHEN06 BASELINE', 'Flat-MARL', '#1f77b4', 0.5),     # Blue
        ('TRAFFIC SHEPHERD', 'Human-Centric MARL', '#ff7f0e', 1.5)  # Orange
    ]

    for i, demand in enumerate([800, 1700]):
        ax = ax1 if i == 0 else ax2
        
        # Plot grouped bars
        for csv_name, display_name, color, offset in model_configs:
            model_data = df[(df['Model'] == csv_name) & (df['Demand (vph/lane)'] == demand)]
            
            means, stds = [], []
            for pr in target_prs:
                row = model_data[model_data['Penetration Rate (%)'] == pr]
                if not row.empty:
                    means.append(row.iloc[0][metric_col + '_mean'])
                    stds.append(row.iloc[0][metric_col + '_std'])
                else:
                    means.append(0.0)
                    stds.append(0.0)
                    
            if any(means):  # Only plot if this model actually has data
                label = display_name if i == 0 else "" # Prevent duplicate legend entries
                ax.bar(x + offset * width, means, width, yerr=stds, capsize=3, label=label, color=color, edgecolor='black', alpha=0.85)
        
        # Handle ALL-HDV Baseline (0% PR Horizontal Line)
        hdv_data = df[(df['Model'] == 'ALL-HDV') & (df['Demand (vph/lane)'] == demand)]
        if not hdv_data.empty:
            hdv_mean = hdv_data.iloc[0][metric_col + '_mean']
            label = 'ALL-HDV (Human Baseline)' if i == 0 else ""
            ax.axhline(y=hdv_mean, color='#7f7f7f', linestyle='--', linewidth=2.5, label=label)
            
        # Formatting
        title_demand = 'Moderate Demand (800 vph)' if demand == 800 else 'Severe Bottleneck (1700 vph)'
        ax.set_title(title_demand, fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pr_labels, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if i == 0:
            ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

    fig.suptitle(f'{title_prefix}', fontsize=16, fontweight='bold', y=1.02)
    
    # Collect legends from ax1 and place them globally
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=11, framealpha=1.0)

    os.makedirs("graphs_output", exist_ok=True)
    out_path = os.path.join("graphs_output", filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved bar chart: {out_path}")
    plt.close()

# ==========================================
# 2. GENERATE TIME-SERIES (.npy) PLOTS
# ==========================================
def plot_time_series_npy(target_metric="speed", demand=1700, pr=20, max_steps=4000):
    """
    Dynamically searches for any available .npy arrays for all models and plots them on a shared timeline.
    """
    npy_models = {
        'DONGCHEN': ('Flat-MARL', '#1f77b4'),
        'SHEPHERD': ('Human-Centric MARL', '#ff7f0e')
    }
    
    plt.figure(figsize=(10, 5))
    plotted_any = False
    
    for prefix, (label, color) in npy_models.items():
        if prefix == 'DONGCHEN':
            npy_file = f"./results_dongchen_baseline/0809/models/{target_metric}_array_{prefix}_{demand}vph_{pr}PR.npy"
        else:
            npy_file = f"./results/MAPPO_Curriculum_seed_0809/models/{target_metric}_array_{prefix}_{demand}vph_{pr}PR.npy"
        if os.path.exists(npy_file):
            data = np.load(npy_file)
            plt.plot(range(max_steps), data, label=label, color=color, alpha=0.9, linewidth=1.5)
            plotted_any = True

    if not plotted_any:
        plt.close()
        return

    if target_metric == "speed":
        ylabel = 'Average Speed (m/s)'
        title = f'System Speed Progression ({demand} vph | {pr}% PR)'
    elif target_metric == "decel":
        ylabel = 'Mean Deceleration Magnitude (m/s²)'
        title = f'Braking Intensity Progression ({demand} vph | {pr}% PR)'
    elif target_metric == "jerk":
        ylabel = 'Mean Abs Jerk (m/s³)'
        title = f'Driving Comfort (Jerk) Progression ({demand} vph | {pr}% PR)'
    else:
        ylabel = 'Value'
        title = f'{target_metric.capitalize()} Progression'

    plt.xlabel('Simulation Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim([0, max_steps])
    
    ticks = np.arange(0, max_steps + 1, 1000)
    plt.xticks(ticks, [f"{int(t/1000)}k" if t > 0 else "0k" for t in ticks], fontsize=11)
    plt.yticks(fontsize=11)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(frameon=True, loc='best', fontsize=11)
    
    os.makedirs("graphs_output", exist_ok=True)
    out_path = os.path.join("graphs_output", f"timeseries_{target_metric}_{demand}vph_{pr}PR.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved time-series chart: {out_path}")
    plt.close()

def main():
    try:
        df, metric_cols = load_and_clean_data("combined_traffic_results.csv")
    except FileNotFoundError as e:
        print(e)
        return
    
    # Comprehensive mapping of ALL metrics present in the CSV dataset to descriptive titles, units, and filenames
    plot_definitions = [
        ('Equivalent Hourly Throughput (veh/h)', 'Network Throughput (Equivalent Hourly Volume)', 'Throughput (vph)\n↑ Higher is Better', 'bar_test_equivalent_throughput.png'),
        ('Raw Arrivals (veh)', 'Raw Arrival Count', 'Vehicle Count\n• Reference Metric', 'bar_raw_arrival_count.png'),
        ('Mean Speed (m/s)', 'Average Network Speed', 'Speed (m/s)\n↑ Higher is Better', 'bar_mean_speed.png'),
        ('Mean Travel Time (s)', 'Mean Travel Time', 'Time (s)\n↓ Lower is Better', 'bar_mean_travel_time.png'),
        ('Collision Rate (%)', 'Collision Rate', 'Collision Rate (%)\n↓ Lower is Better', 'bar_collision_rate.png'),
        ('Avg Near-Collisions', 'Near-Collision Events', 'Count per Episode\n↓ Lower is Better', 'bar_near_collision_count.png'),
        ('Avg TTC Violations', 'Time-To-Collision (TTC) Violations', 'Violations per Episode\n↓ Lower is Better', 'bar_ttc_violations.png'),
        ('Avg Hard Braking', 'Hard Braking Events', 'Count per Episode\n↓ Lower is Better', 'bar_hard_braking_count.png'),
        ('Acceleration Var (m/s^2)', 'Longitudinal Acceleration Variance', 'Variance ((m/s²)²)\n↓ Lower is Better', 'bar_acceleration_variance.png'),
        ('Mean Abs Jerk (m/s^3)', 'Driving Comfort: Mean Absolute Jerk', 'Jerk (m/s³)\n↓ Lower is Better', 'bar_mean_abs_jerk.png'),
        ('Wave Intensity (Var)', 'Wave Interruption Frequency', 'Frequency\n↓ Lower is Better', 'bar_wave_interruption_frequency.png'),
        ('Raw Interventions (Sum)', 'Raw Intervention Count', 'Count\n↓ Lower is Better', 'bar_raw_intervention_count.png'),
        ('Interventions Per CAV', 'Intervention Duration', 'Duration (s)\n↓ Lower is Better', 'bar_intervention_duration.png'),
        ('Duty Cycle (%)', 'Control Efficiency: CAV Active Duty Cycle', 'Duty Cycle (%)\n↓ Lower is Better', 'bar_duty_cycle.png'),
        ('Mean Decel Magnitude (m/s^2)', 'Mean Deceleration Magnitude', 'Deceleration (m/s²)\n↓ Lower is Better', 'bar_mean_decel_magnitude.png'),
        ('Mean Duration (s)', 'Mean Episode/Maneuver Duration', 'Time (s)\n• Reference Metric', 'bar_mean_duration.png'),
        ('Success Rate (%)', 'Ramp Merging Success Rate', 'Success Rate (%)\n↑ Higher is Better', 'bar_success_rate.png'),
        ('Avg Max Queue (veh)', 'Avg Maximum Queue Length', 'Vehicles (veh)\n↓ Lower is Better', 'bar_avg_max_queue.png'),
        ('Absolute Max Queue (veh)', 'Absolute Maximum Queue Length', 'Vehicles (veh)\n↓ Lower is Better', 'bar_absolute_max_queue.png'),
        ('Mean Delay (s)', 'Mean Delay Duration', 'Time (s)\n↓ Lower is Better', 'bar_mean_delay.png'),
        ('90th-Percentile Delay (s)', '90th-Percentile Tail Delay', 'Delay (s)\n↓ Lower is Better', 'bar_90th_percentile_delay.png')
    ]
    
    print("Generating Comprehensive Bar Charts for Every Metric...")
    for col, title, ylabel, filename in plot_definitions:
        if col in metric_cols:
            plot_comparative_bar_chart(df, col, title, ylabel, filename)
        else:
            print(f"  [Skip] Metric '{col}' not found in CSV columns.")

    print("\nAttempting to generate Time-Series NPY charts...")
    for target in ["speed", "decel", "jerk"]:
        for demand in [800, 1700]:
            for pr in [5, 10, 20]:
                plot_time_series_npy(target_metric=target, demand=demand, pr=pr)

    print("\nGraph generation complete. Check the 'graphs_output' directory.")

if __name__ == "__main__":
    main()