import os
import numpy as np
import matplotlib.pyplot as plt

def smooth(x, timestamps=9):
    """Applies a moving average filter to smooth the data lines."""
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - timestamps)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y

def plot_training_progress(file_path='./results/episode_rewards.npy'):
    """
    Plots the training progress of the Traffic Shepherd MA2C agent.
    """
    if not os.path.exists(file_path):
        print(f"Error: Could not find data file at {file_path}")
        print("Ensure your MA2C training script saves 'agent.episode_rewards' as a .npy file.")
        return

    # Load the rewards array
    rewards = np.load(file_path)
    
    plt.figure(figsize=(8, 5))
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Human-Centric Reward", fontsize=12)
    plt.title("Traffic Shepherd: MA2C Training Progress", fontsize=14)
    
    # Plot raw rewards in the background with high transparency
    plt.plot(rewards, alpha=0.3, color='#1f77b4', label='Raw Reward')
    
    # Plot the smoothed trendline over the top
    smoothed_rewards = smooth(rewards, timestamps=20)
    plt.plot(smoothed_rewards, color='#1f77b4', linewidth=2, label='Smoothed Trend')
    
    # Formatting
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # You can pass a specific path here if you save multiple runs, 
    # e.g., plot_training_progress('./results/run_1_rewards.npy')
    plot_training_progress()