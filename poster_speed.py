import matplotlib.pyplot as plt
import numpy as np
import os

def plot_speed_over_time(ts_file, dc_file, output_filename):
    if not os.path.exists(ts_file) or not os.path.exists(dc_file):
        print(f"Error: Could not find the speed_array .npy files. Check your paths.")
        return

    # Load the speed arrays
    ts_speed = np.load(ts_file)
    dc_speed = np.load(dc_file)
    
    # Generate an X-axis for the steps
    steps = np.arange(len(ts_speed))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot both lines with distinct colors and thick lines for poster visibility
    ax.plot(steps, dc_speed, label='Flat MARL', color='#1f77b4', linewidth=2.5, alpha=0.9)
    ax.plot(steps, ts_speed, label='Human-Centric MARL', color='#ff7f0e', linewidth=2.5, alpha=0.9)
    
    # Formatting for Poster Readability
    ax.set_xlabel('Simulation Step', fontsize=14, fontweight='bold')
    ax.set_ylabel('Network Average Speed (m/s)', fontsize=14, fontweight='bold')
    ax.set_title('Network Speed Stability Over Time (1700 vph)', fontsize=16, pad=15)
    
    # Format the X-axis ticks to show '1k, 2k, 3k' instead of raw numbers
    max_steps = len(ts_speed)
    ticks = np.arange(0, max_steps + 1, 1000)
    labels = [f"{int(t/1000)}k" if t > 0 else "0" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=12)
    plt.yticks(fontsize=12)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Place legend cleanly outside the dense data area if possible
    ax.legend(loc='lower left', fontsize=13, framealpha=1.0)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Successfully saved: {output_filename}")
    plt.close()

if __name__ == "__main__":
    # =========================================================================
    # EDIT THESE PATHS TO POINT TO YOUR ACTUAL speed_array .npy FILES
    # Example: "results/speed_array_DONGCHEN_1700vph_20PR.npy"
    # =========================================================================
    
    TS_SPEED_FILE = "./results/real_TS/models/speed_array_SHEPHERD_1700vph_10PR.npy"
    DC_SPEED_FILE = "./results_dongchen_baseline/REAL_FLAT/models/speed_array_DONGCHEN_1700vph_10PR.npy"

    print("Generating Speed Time-Series Plot...")
    
    plot_speed_over_time(
        ts_file=TS_SPEED_FILE, 
        dc_file=DC_SPEED_FILE, 
        output_filename="Poster_Speed_Time_Series.png"
    )