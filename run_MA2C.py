import os
import torch as th
import numpy as np

from on_ramp_env import MARLOnRampEnv
from MA2C import TrafficShepherdMA2C

# ---------------------------------------------------------
# 1. Centralized Configuration
# ---------------------------------------------------------
CONFIG = {
    "experiment_name": "Traffic_Shepherd_Baseline",
    "random_seed": 2026,
    
    # Environment Settings
    "max_steps": 500,
    "use_gui": False,
    
    # MA2C Agent Settings
    "actor_lr": 0.001,
    "critic_lr": 0.001,
    "batch_size": 128,
    "memory_capacity": 20000,
    "num_episodes": 100,
    
    # Traffic Shepherd Reward Weights[cite: 5]
    "w_efficiency": 1.0,
    "w_safety": 4,
    "w_stability": 1.0,
    "w_fairness": 1.0,
    "w_control_cost": 1.0
}

def set_random_seeds(seed):
    """Ensures experiments are reproducible, adapted from original MA2C code."""
    th.manual_seed(seed)
    np.random.seed(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    print(f"Starting Experiment: {CONFIG['experiment_name']}")
    
    # Apply the random seed
    set_random_seeds(CONFIG["random_seed"])

    print("Initializing Traffic Shepherd Environment...")
    env = MARLOnRampEnv(
        net_file='SUMO_network/test.net.xml',
        route_file='SUMO_network/highway_onramp_actual.rou.xml',
        use_gui=CONFIG["use_gui"], 
        max_steps=CONFIG["max_steps"]
    )

    print("Initializing Custom MA2C Agent...")
    agent = TrafficShepherdMA2C(
        env=env,
        state_dim=4, 
        action_dim=5, 
        memory_capacity=CONFIG["memory_capacity"],
        max_steps=CONFIG["max_steps"],
        actor_lr=CONFIG["actor_lr"],
        critic_lr=CONFIG["critic_lr"],
        batch_size=CONFIG["batch_size"],
        use_cuda=False 
    )

    print("Starting MA2C training loop...")
    
    for episode in range(CONFIG["num_episodes"]):
        agent.explore()
        agent.train()
        
        latest_reward = agent.episode_rewards[-1] if agent.episode_rewards else 0.0
        print(f"Training Episode {episode + 1}/{CONFIG['num_episodes']} | Total Reward: {latest_reward:.2f}")

    print("Training finished successfully.")
    
    # Save the reward history for the plotting script
    os.makedirs("./results", exist_ok=True)
    np.save(f"./results/{CONFIG['experiment_name']}_rewards.npy", np.array(agent.episode_rewards))
    
    env.close()

if __name__ == "__main__":
    main()