from z_merge_env import ZMergeEnv
from z_merge_agent import ZMergeAgent

def main():
    # Notice: use_gui=True so you can visually watch what the network learned!
    env = ZMergeEnv("SUMO_network/test.net.xml", "SUMO_network/highway_onramp_actual.rou.xml", use_gui=True)
    
    # Initialize the agent (must match the training dimensions)
    agent = ZMergeAgent(state_dim=22) 
    
    # Load your saved 200-episode brain
    try:
        agent.load("zmerge_v1")
        print("Successfully loaded trained models!")
    except FileNotFoundError:
        print("Error: Could not find the saved models. Ensure 'zmerge_v1_qnet.pth' is in the folder.")
        return
    
    episodes = 5
    max_steps = 1000
    
    # THE MOST IMPORTANT LINE: Epsilon is strictly 0.0 (Pure Exploitation)
    epsilon = 0.0 
    
    print("Starting Z-Merge Evaluation Pipeline...")
    
    for ep in range(episodes):
        obs_dict = env.reset()
        episode_reward = 0.0
        
        for step in range(max_steps):
            actions_dict = {}
            
            # Select actions for all active agents (No randomness)
            for agent_id, state in obs_dict.items():
                d_action, c_params = agent.act(state, epsilon)
                actions_dict[agent_id] = (d_action, c_params)
                
            # Step environment
            next_obs_dict, rewards_dict, dones_dict = env.step(actions_dict)
            
            # Tally rewards (NOTICE: We completely removed agent.train_step() and memory.push())
            for agent_id in actions_dict.keys():
                if agent_id in next_obs_dict: 
                    episode_reward += rewards_dict[agent_id]
                    
            obs_dict = next_obs_dict
            
        print(f"Evaluation Episode {ep+1}/{episodes} | Total Reward: {episode_reward:.2f}")

if __name__ == "__main__":
    main()