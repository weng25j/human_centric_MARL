import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_poster_success_vs_intervention():
    # Data correctly mapped from image_3e65dc.png (Demand 1700)
    # Format: [Mean Duration (s), Success Rate (%), Avg TTC Violations, Label]
    
    data = {
        'ALL-HDV': {
            'color': '#7f7f7f', 'marker': '*', 
            'points': [(0.00, 23.36, 373.9, '0% PR')]
        },
        'RULE-CAV': {
            'color': '#d62728', 'marker': 's', 
            'points': [(0.90, 34.25, 1028.7, '5% PR'),
                       (1.04, 36.94, 1218.5, '10% PR'),
                       (1.25, 42.09, 1146.8, '20% PR')]
        },
        'SELFISH-CAV': {
            'color': '#9467bd', 'marker': 'D', 
            'points': [(6.64, 25.22, 368.5, '5% PR'),
                       (6.58, 25.45, 399.6, '10% PR'),
                       (6.62, 32.09, 348.4, '20% PR')]
        },
        'Flat-MARL': {
            'color': '#1f77b4', 'marker': 'o', 
            'points': [(0.21, 37.61, 489.2, '5% PR'),
                       (0.32, 36.94, 467.7, '10% PR'),
                       (0.53, 35.45, 496.2, '20% PR')]
        },
        'Human-Centric MARL (Ours)': {
            'color': '#ff7f0e', 'marker': '^', 
            'points': [(2.03, 26.49, 365.1, '5% PR'),
                       (2.50, 24.18, 403.2, '10% PR'),
                       (2.57, 31.27, 371.1, '20% PR')]
        }
    }

    fig, ax = plt.subplots(figsize=(11, 7))

    # Add the "Fast & Safe Cluster" highlighted background rectangle
    cluster_rect = patches.Rectangle((-0.5, 20), 4.0, 30, linewidth=2, edgecolor='#2ca02c', 
                                     facecolor='#eaffea', linestyle='--', alpha=0.5, zorder=0)
    ax.add_patch(cluster_rect)
    
    ax.text(-0.2, 48, '— Short Duration (Efficient)\n— Fast & Safe Cluster (Ideal for MARL)', 
            color='#2ca02c', fontsize=11, fontweight='bold', va='top')

    legend_handles = []

    for model_name, props in data.items():
        color = props['color']
        marker = props['marker']
        added_to_legend = False
        
        for duration, success, ttc, label in props['points']:
            # Scale bubble size down slightly
            bubble_size = ttc * 0.4
            
            scatter = ax.scatter(duration, success, s=bubble_size, c=color, marker=marker,
                                 edgecolors='black', linewidth=1.2, alpha=0.8, zorder=3)
            
            # Annotate PR percentage next to the bubble
            if model_name == 'ALL-HDV':
                ax.text(duration + 0.15, success, label, fontsize=9, va='center', zorder=4)
            else:
                ax.text(duration + 0.15, success - 0.5, label, fontsize=9, va='center', zorder=4)
                
            if not added_to_legend:
                proxy = plt.scatter([], [], s=100, c=color, marker=marker, edgecolors='black', label=model_name)
                legend_handles.append(proxy)
                added_to_legend = True

    # Formatting axes and limits
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(10, 50)
    
    ax.set_xlabel('Control Inefficiency → Mean Intervention Duration (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Merge Performance → Merging Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Control Efficiency vs. Merge Performance vs. Bottleneck Safety\nBubble Size = Avg TTC Violations (Demand 1700 vph)', 
                 fontsize=14, fontweight='bold', pad=15)
                 
    ax.set_yticks(np.arange(10, 55, 5))
    ax.set_yticklabels([f"{y:.1f}%" for y in np.arange(10, 55, 5)], fontsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)

    # Primary legend
    model_legend = ax.legend(handles=legend_handles, loc='center right', title="Models", 
                             bbox_to_anchor=(0.98, 0.75), framealpha=1.0, title_fontsize=12, fontsize=10)
    ax.add_artist(model_legend)

    # Secondary legend for sizes
    size_400 = plt.scatter([], [], s=400 * 0.4, c='white', edgecolors='black')
    size_1200 = plt.scatter([], [], s=1200 * 0.4, c='white', edgecolors='black')
    
    size_legend = plt.legend([size_400, size_1200], ['400', '1,200'], 
                             loc='lower right', title="Avg TTC Violations\n(Smaller is Safer)", 
                             labelspacing=1.5, borderpad=1.2, bbox_to_anchor=(0.98, 0.05))

    plt.tight_layout()
    plt.savefig('Poster_Success_vs_Duration.png', dpi=300)
    print("Saved 'Poster_Success_vs_Duration.png'")

if __name__ == "__main__":
    plot_poster_success_vs_intervention()