import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

def smooth(x, timestamps=9):
    """Applies a moving average filter to smooth the data lines."""
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y

def load_and_aggregate_data(base_path, policy_name, seeds, obs_window):
    """
    Loads .npy files for multiple random seeds of a specific policy,
    smooths them, and calculates the mean and standard deviation.
    """
    data_list = []
    for seed in seeds:
        file_path = os.path.join(base_path, f"{policy_name}_seed_{seed}.npy")
        if os.path.exists(file_path):
            raw_data = np.load(file_path)
            # Ensure the data matches our observation window, then smooth it
            smoothed_data = smooth(raw_data[:obs_window])
            data_list.append(smoothed_data)
        else:
            print(f"Warning: Data file not found -> {file_path}")
            
    if not data_list:
        # Return zeros if no data is found so the plot doesn't crash
        return np.zeros(obs_window), np.zeros(obs_window)
        
    stacked_data = np.vstack(data_list)
    mean_data = np.mean(stacked_data, axis=0)
    std_data = np.std(stacked_data, axis=0)
    
    return mean_data, std_data

def main():
    # ---------------------------------------------------------
    # 1. Configuration & Plotting Aesthetics
    # ---------------------------------------------------------
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    sns.set_color_codes()
    
    # Custom color palette for the 5 Traffic Shepherd policies
    colors = sns.color_palette("husl", 5) 
    alpha = 0.3
    legend_size = 14
    line_size = 2.5
    tick_size = 14
    label_size = 16

    # ---------------------------------------------------------
    # 2. Data Loading Setup
    # ---------------------------------------------------------
    results_dir = "./results" # The folder where your .npy files are saved
    obs_window = 100          # Number of evaluation epochs to plot
    seeds = [0, 1000, 2026]   # The random seeds used in your experiments[cite: 5]
    
    X = np.arange(obs_window)
    
    # The 5 policies defined in the Traffic Shepherd research plan[cite: 5]
    policies = {
        "All-HDV":        {"name_in_file": "all_hdv",        "color": colors[0], "linestyle": "--"},
        "Rule-CAV":       {"name_in_file": "rule_cav",       "color": colors[1], "linestyle": "-."},
        "Selfish-CAV":    {"name_in_file": "selfish_cav",    "color": colors[2], "linestyle": ":"},
        "Flat-MARL":      {"name_in_file": "flat_marl",      "color": colors[3], "linestyle": "-"},
        "Traffic-Shepherd":{"name_in_file": "traffic_shep",  "color": colors[4], "linestyle": "-"}
    }

    # ---------------------------------------------------------
    # 3. Plotting
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Evaluation Rewards across Policy Classes', size=label_size+2)

    for display_name, config in policies.items():
        mean_data, std_data = load_and_aggregate_data(
            results_dir, 
            config["name_in_file"], 
            seeds, 
            obs_window
        )
        
        lower_bound = mean_data - std_data
        upper_bound = mean_data + std_data

        # Plot the mean line
        ax.plot(X, mean_data, lw=line_size, label=display_name, 
                linestyle=config["linestyle"], color=config["color"])
        
        # Fill the standard deviation shading
        ax.fill_between(X, lower_bound, upper_bound, 
                        facecolor=config["color"], edgecolor='none', alpha=alpha)

    # ---------------------------------------------------------
    # 4. Axes Formatting & Output
    # ---------------------------------------------------------
    leg = ax.legend(fontsize=legend_size, loc='lower right', ncol=2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)

    ax.set_xlim(0, obs_window - 1)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_xlabel('Evaluation Epochs', fontsize=label_size)
    ax.set_ylabel('Human-Centric Reward', fontsize=label_size)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig("traffic_shepherd_evaluation.pdf")
    print("Plot saved successfully to 'traffic_shepherd_evaluation.pdf'")
    plt.show()

if __name__ == "__main__":
    main()
    