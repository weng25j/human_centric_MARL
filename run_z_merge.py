import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from z_merge_env import ZMergeEnv
from z_merge_agent import ZMergeAgent

def main():
    # --- 1. SETUP OUTPUT DIRECTORY ---
    base_dir = "./results_zmerge/"
    # Get current time for folder naming (matching your other scripts)
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = os.path.join(base_dir, now)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup Environment and Agent
    env = ZMergeEnv("SUMO_network/test.net.xml", "SUMO_network/highway_onramp_actual.rou.xml", use_gui=False)
    agent = ZMergeAgent(state_dim=22) 

    epsilon = 0.5

    try:
        agent.load("zmerge_v1")
    except FileNotFoundError:
        print("No saved models found, starting from scratch.")
        epsilon = 0.99
    
    episodes = 200
    max_steps = 1000
    batch_size = 128
    
    epsilon_decay = 0.995
    epsilon_min = 0.05
    
    reward_scale = 20.0 
    
    # --- 2. INITIALIZE TRACKING LIST ---
    eval_rewards = []
    
    print(f"Starting Z-Merge Hybrid Training Pipeline... Saving to {output_dir}")
    
    for ep in range(episodes):
        obs_dict = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            actions_dict = {}
            
            # Select actions for all active agents
            for agent_id, state in obs_dict.items():
                d_action, c_params = agent.act(state, epsilon)
                actions_dict[agent_id] = (d_action, c_params)
                
            # Step environment
            next_obs_dict, rewards_dict, dones_dict = env.step(actions_dict)
            
            # Store transitions in shared Replay Buffer
            for agent_id in actions_dict.keys():
                if agent_id in next_obs_dict: 
                    d_act, c_params = actions_dict[agent_id]
                    
                    # Scale reward for neural network stability
                    scaled_reward = rewards_dict[agent_id] / reward_scale
                    
                    agent.memory.push(
                        obs_dict[agent_id], 
                        d_act, 
                        c_params, 
                        scaled_reward, 
                        next_obs_dict[agent_id], 
                        dones_dict[agent_id]
                    )
                    
                    # Accumulate raw reward for fair visual comparison
                    episode_reward += rewards_dict[agent_id]
                    
            # Train Networks
            agent.train_step(batch_size)
            
            obs_dict = next_obs_dict
            
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {ep+1}/{episodes} | Total Reward: {episode_reward:.2f} | Epsilon: {epsilon:.2f}")
        
        # --- 3. STORE EPISODE DATA ---
        eval_rewards.append(episode_reward)

    print("Training complete! Saving agent and data...")
    agent.save(os.path.join(output_dir, "zmerge_v1"))

    # --- 4. SAVE RAW DATA AND PLOT ---
    # Save the numpy array so it can be loaded by your comparison plotter later
    np.save(os.path.join(output_dir, "eval_rewards.npy"), eval_rewards)
    
    # Generate and save the standalone learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episodes + 1), eval_rewards, color='green', linewidth=2, label="Z-Merge Baseline")
    plt.title("Z-Merge Agent Performance", fontsize=14, fontweight='bold')
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Human-Centric Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zmerge_learning_curve.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    main()