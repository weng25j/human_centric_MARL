import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import traci
import random
import torch

from z_merge_env import ZMergeEnv
from z_merge_agent import ZMergeAgent
# Assuming common.utils is available in your workspace as in the MAPPO script
try:
    from common.utils import agg_double_list, init_dir
except ImportError:
    # Fallback if utils are missing
    def init_dir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
        return {'models': os.path.join(output_dir, "models")}

CONFIG = {
    'state_dim': 22,
    'MAX_EPISODES': 3000,          
    'EPISODES_BEFORE_TRAIN': 10,    
    'ROLL_OUT_N_STEPS': 1000,       # Replaced 4000 with Z-Merge's 1000
    'EVAL_INTERVAL': 50,            
    'EVAL_EPISODES': 10,            
    'BATCH_SIZE': 128,              # From Z-Merge
    'MEMORY_CAPACITY': 50000,
    'epsilon_start': 0.99,
    'epsilon_decay': 0.995,
    'epsilon_min': 0.05,
    'reward_scale': 20.0
}

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    default_base_dir = "./results_zmerge/"
    parser = argparse.ArgumentParser(description='Train or evaluate policy on SUMO RL environment using Z-Merge')
    parser.add_argument('--base-dir', type=str, default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, default='train', help="train or evaluate")
    parser.add_argument('--model-dir', type=str, default='', help="pretrained model path")
    parser.add_argument('--gui', action='store_true', help="Run SUMO with GUI (sumo-gui)")
    parser.add_argument('--start-episode', type=int, default=0, help="Episode to resume training from")
    parser.add_argument('--seed', type=int, default=42, help="Global random base seed")
    args = parser.parse_args()
    return args

def generate_mixed_traffic_route(target_penetration_rate, lane_demand_vph, num_mainline_lanes=2):
    total_mainline_vph = lane_demand_vph * num_mainline_lanes
    ramp_vph = lane_demand_vph * 0.35 

    cav_main_vph = total_mainline_vph * target_penetration_rate
    hdv_main_vph = total_mainline_vph * (1.0 - target_penetration_rate)
    
    cav_ramp_vph = ramp_vph * target_penetration_rate
    hdv_ramp_vph = ramp_vph * (1.0 - target_penetration_rate)

    flows = []
    if hdv_main_vph > 0:
        flows.append(f'<flow id="human_flow" type="hdv_mixture" route="route_main" begin="0" end="10000" vehsPerHour="{hdv_main_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>')
    if cav_main_vph > 0:
        flows.append(f'<flow id="cav_main_flow" type="cav" route="route_main" begin="0" end="10000" vehsPerHour="{cav_main_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>')
    if hdv_ramp_vph > 0:
        flows.append(f'<flow id="human_ramp_flow" type="hdv_mixture" route="route_ramp" begin="0" end="10000" vehsPerHour="{hdv_ramp_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>')
    if cav_ramp_vph > 0:
        flows.append(f'<flow id="cav_ramp_flow" type="cav" route="route_ramp" begin="0" end="10000" vehsPerHour="{cav_ramp_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>')
        
    flows_xml = "\n    ".join(flows)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vTypeDistribution id="hdv_mixture">
        <vType id="hdv_cautious" probability="0.2" carFollowModel="IDM" accel="2.0" decel="4.5" emergencyDecel="7.5" apparentDecel="4.5" tau="1.8" speedFactor="0.9" speedDev="0.05" lcCooperative="2.0" lcAssertive="0.1" color="0,255,0"/>
        <vType id="hdv_normal" probability="0.6" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" apparentDecel="4.5" tau="1.2" speedFactor="1.0" speedDev="0.1" lcCooperative="1.0" lcAssertive="1.0" color="255,255,255"/>
        <vType id="hdv_aggressive" probability="0.2" carFollowModel="IDM" accel="3.5" decel="4.5" emergencyDecel="9.0" apparentDecel="4.5" tau="0.8" speedFactor="1.25" speedDev="0.1" lcCooperative="0.0" lcAssertive="5.0" color="0,0,255"/>
    </vTypeDistribution>
    <vType id="cav" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" tau="1.0" color="255,0,0"/>
    <route id="route_main" edges="E0 E1 E6" />
    <route id="route_ramp" edges="E_ramp E1 E6" />

    {flows_xml}
</routes>
"""
    os.makedirs("SUMO_network", exist_ok=True)
    with open("SUMO_network/highway_onramp_z_merge.rou.xml", "w") as f:
        f.write(xml_content)


class TraCIEnvWrapper:
    """Safely wraps the Z-Merge environment to handle SUMO/TraCI crashes and inactive vehicles."""
    def __init__(self, net_file, route_file, use_gui):
        self.env = ZMergeEnv(net_file, route_file, use_gui=use_gui)
        
    def reset(self):
        try:
            return self.env.reset()
        except Exception:
            return {}
            
    def step(self, action_dict):
        try:
            active_vehs = set(traci.vehicle.getIDList())
        except Exception:
            active_vehs = set()
            
        safe_action_dict = {cav: act for cav, act in action_dict.items() if cav in active_vehs}
        
        try:
            next_obs_dict, rewards_dict, dones_dict = self.env.step(safe_action_dict)
        except Exception:
            next_obs_dict, rewards_dict, dones_dict = {}, {}, {}
            
        # Ensure we return entries matching the original requested actions
        final_rewards = {cav: rewards_dict.get(cav, 0.0) for cav in action_dict.keys()}
        final_dones = {cav: dones_dict.get(cav, True) for cav in action_dict.keys()}
        
        return next_obs_dict, final_rewards, final_dones


def train(args):
    set_global_seeds(args.seed)
    base_dir = args.base_dir

    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = f"{base_dir}ZMerge_Curriculum_seed_{args.seed}_{now}/"
    dirs = init_dir(output_dir)

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_z_merge.rou.xml"

    env_wrapper = TraCIEnvWrapper(net_file, route_file, use_gui=args.gui)
    agent = ZMergeAgent(state_dim=CONFIG['state_dim']) 

    epsilon = CONFIG['epsilon_start']

    if args.model_dir and os.path.exists(args.model_dir):
        print(f"Loading pre-trained model from {args.model_dir} to resume training...")
        agent.load(args.model_dir)
        epsilon = CONFIG['epsilon_min'] # Assume fully trained behavior
    else:
        print("Beginning Z-Merge Curriculum Training from scratch...")

    n_episodes = args.start_episode
    if n_episodes > 0:
        print(f"Manually setting start episode to {n_episodes}...")

    eval_rewards = []
    episode_rewards_history = []
    print(f"Beginning Z-Merge Curriculum Training over SUMO TraCI (Base Seed: {args.seed})...")

    while n_episodes < CONFIG['MAX_EPISODES']:
        # --- Curriculum Setup ---
        if n_episodes < 1000:
            train_demand = 800     
            train_pr = 0.20  
        elif n_episodes < 2000:
            train_demand = random.choice([800, 1200]) 
            train_pr = 0.10  
        else:
            train_demand = random.choice([800, 1200, 1700]) 
            train_pr = random.choice([0.05, 0.10, 0.20])

        generate_mixed_traffic_route(target_penetration_rate=train_pr, lane_demand_vph=train_demand)

        episode_seed = args.seed + n_episodes 
        
        traci_args = [
            sumo_binary, "-n", net_file, "-r", route_file, 
            "--seed", str(episode_seed), 
            "--start", "--quit-on-end",
            "--step-length", "0.1",                  
            "--no-step-log", "true",
            "--collision.action", "remove",
            "--collision.check-junctions", "true", 
            "--collision.mingap-factor", "0.0"
        ]
        traci.start(traci_args)
        
        # --- Episode Interaction Loop ---
        episode_reward = 0.0
        try:
            obs_dict = env_wrapper.reset()
            
            for step in range(CONFIG['ROLL_OUT_N_STEPS']):
                actions_dict = {}
                
                for agent_id, state in obs_dict.items():
                    d_action, c_params = agent.act(state, epsilon)
                    actions_dict[agent_id] = (d_action, c_params)
                    
                next_obs_dict, rewards_dict, dones_dict = env_wrapper.step(actions_dict)
                
                for agent_id in actions_dict.keys():
                    if agent_id in next_obs_dict: 
                        d_act, c_params = actions_dict[agent_id]
                        scaled_reward = rewards_dict[agent_id] / CONFIG['reward_scale']
                        
                        agent.memory.push(
                            obs_dict[agent_id], 
                            d_act, 
                            c_params, 
                            scaled_reward, 
                            next_obs_dict[agent_id], 
                            dones_dict[agent_id]
                        )
                        episode_reward += rewards_dict[agent_id]
                
                # Train network if we've passed the initial collection phase
                if n_episodes >= CONFIG['EPISODES_BEFORE_TRAIN'] and len(agent.memory) > CONFIG['BATCH_SIZE']:
                    agent.train_step(CONFIG['BATCH_SIZE'])
                
                obs_dict = next_obs_dict

        except Exception as e:
            print(f"  [Warning] Episode {n_episodes} aborted early due to SUMO crash. Resetting...")
            
        try:
            traci.close()
        except Exception:
            pass

        # --- Post-Episode Tracking ---
        epsilon = max(CONFIG['epsilon_min'], epsilon * CONFIG['epsilon_decay'])
        episode_rewards_history.append(episode_reward)
        n_episodes += 1
            
        if (n_episodes) % CONFIG['EVAL_INTERVAL'] == 0:
            trailing_avg = np.mean(episode_rewards_history[-CONFIG['EVAL_INTERVAL']:])
            print(f"Episode {n_episodes} | Avg Reward: {trailing_avg:.2f} | Epsilon: {epsilon:.2f} | Demand: {train_demand} vph/lane | PR: {train_pr:.2f}")
            eval_rewards.append(trailing_avg)
            agent.save(os.path.join(dirs['models'], f"zmerge_ep_{n_episodes}"))

    # Final Save
    agent.save(os.path.join(dirs['models'], "zmerge_final"))
    np.save(os.path.join(output_dir, "eval_rewards.npy"), eval_rewards)

    # Plot
    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel(f"Evaluation Intervals (x{CONFIG['EVAL_INTERVAL']})")
    plt.ylabel("Average Z-Merge Reward")
    plt.title(f"Z-Merge Curriculum Performance (Seed {args.seed})")
    plt.savefig(os.path.join(output_dir, "training_curve.png"), dpi=300)
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train(args)