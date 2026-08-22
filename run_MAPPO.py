import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import traci
import random
import torch

from MAPPO import MAPPO
from traffic_shepherd import TrafficShepherdEnv
from common.utils import agg_double_list, init_dir

CONFIG = {
    'actor_hidden_size': 256,      
    'critic_hidden_size': 256,      
    'MAX_EPISODES': 3000,          
    'EPISODES_BEFORE_TRAIN': 10,    
    'ROLL_OUT_N_STEPS': 4000,
    'EVAL_INTERVAL': 50,            
    'EVAL_EPISODES': 10,            
    'BATCH_SIZE': 256,
    'MEMORY_CAPACITY': 50000,
    'reward_gamma': 0.99,
    'MAX_GRAD_NORM': 0.5,
    'ENTROPY_REG': 0.01,
    'reward_type': 'global_R',
    'TARGET_UPDATE_STEPS': 5,
    'TARGET_TAU': 0.01,
    'actor_lr': 3e-4,              
    'critic_lr': 1e-3,              
    'reward_scale': 20.0
}

def set_global_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_args():
    default_base_dir = "./results/"
    parser = argparse.ArgumentParser(description='Train or evaluate policy on SUMO RL environment using MAPPO')
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
    with open("SUMO_network/highway_onramp_actual_shepherd.rou.xml", "w") as f:
        f.write(xml_content)

class TraCIEnvWrapper:
    def __init__(self, net_file, route_file, num_cavs):
        self.env = TrafficShepherdEnv(net_file, route_file, num_cavs=num_cavs)
        
    def reset(self):
        try:
            return self.env.get_observations({})
        except Exception:
            return {}, []
            
    def step(self, action_dict):
        try:
            active_vehs = set(traci.vehicle.getIDList())
        except Exception:
            active_vehs = set()
            
        safe_action_dict = {cav: act for cav, act in action_dict.items() if cav in active_vehs}
        
        try:
            rewards = self.env.step(safe_action_dict)
        except Exception:
            rewards = {}
            
        final_rewards = {cav: rewards.get(cav, 0.0) for cav in action_dict.keys()}
        return final_rewards
        
    def get_observations(self, dummy):
        try:
            obs_dict, active_cavs = self.env.get_observations(dummy)
            active_vehs = set(traci.vehicle.getIDList())
            
            safe_obs = {cav: obs for cav, obs in obs_dict.items() if cav in active_vehs}
            safe_active = [cav for cav in active_cavs if cav in active_vehs]
            return safe_obs, safe_active
        except Exception:
            return {}, []

def train(args):
    set_global_seeds(args.seed)
    base_dir = args.base_dir

    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = f"{base_dir}MAPPO_Curriculum_seed_0809/"
    dirs = init_dir(output_dir)

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual_shepherd.rou.xml"
    NUM_CAVS = 5
    
    state_dim = 14  
    action_dim = 3  

    env_wrapper = TraCIEnvWrapper(net_file, route_file, num_cavs=NUM_CAVS)

    mappo = MAPPO(env=env_wrapper, memory_capacity=CONFIG['MEMORY_CAPACITY'],
                  state_dim=state_dim, action_dim=action_dim, num_cavs=NUM_CAVS,
                  batch_size=CONFIG['BATCH_SIZE'], entropy_reg=CONFIG['ENTROPY_REG'],
                  roll_out_n_steps=CONFIG['ROLL_OUT_N_STEPS'],
                  actor_hidden_size=CONFIG['actor_hidden_size'], critic_hidden_size=CONFIG['critic_hidden_size'],
                  actor_lr=CONFIG['actor_lr'], critic_lr=CONFIG['critic_lr'], reward_scale=CONFIG['reward_scale'],
                  target_update_steps=CONFIG['TARGET_UPDATE_STEPS'], target_tau=CONFIG['TARGET_TAU'],
                  reward_gamma=CONFIG['reward_gamma'], reward_type=CONFIG['reward_type'],
                  max_grad_norm=CONFIG['MAX_GRAD_NORM'], episodes_before_train=CONFIG['EPISODES_BEFORE_TRAIN'])

    if args.model_dir and os.path.exists(args.model_dir):
        print(f"Loading pre-trained model from {args.model_dir} to resume training...")
        mappo.load(args.model_dir, train_mode=True)
    else:
        print("Beginning Traffic Shepherd Curriculum Training from scratch...")

    if args.start_episode > 0:
        mappo.n_episodes = args.start_episode
        print(f"Manually setting start episode to {mappo.n_episodes}...")

    eval_rewards = []
    print(f"Beginning MAPPO Curriculum Training over SUMO TraCI (Base Seed: {args.seed})...")

    while mappo.n_episodes < CONFIG['MAX_EPISODES']:
        if mappo.n_episodes < 1000:
            train_demand = 800     
            train_pr = 0.20  
        elif mappo.n_episodes < 2000:
            train_demand = random.choice([800, 1200]) 
            train_pr = 0.10  
        else:
            train_demand = random.choice([800, 1200, 1700]) 
            train_pr = random.choice([0.05, 0.10, 0.20])

        generate_mixed_traffic_route(target_penetration_rate=train_pr, lane_demand_vph=train_demand)

        episode_seed = args.seed + mappo.n_episodes 
        
        traci_args = [
            sumo_binary, "-n", net_file, "-r", route_file, 
            "--seed", str(episode_seed), 
            "--start", "--quit-on-end",
            "--step-length", "0.1",                  # <-- ADD THIS LINE
            "--no-step-log", "true",
            "--collision.action", "remove",
            "--collision.check-junctions", "true", 
            "--collision.mingap-factor", "0.0"
        ]
        traci.start(traci_args)
        
        # FIX: Try/except block gracefully ends the episode if SUMO crashes, allowing training to continue
        try:
            mappo.obs_dict, mappo.active_cavs = env_wrapper.get_observations({})
            mappo.n_steps = 0
            mappo.interact()
        except Exception as e:
            print(f"  [Warning] Episode {mappo.n_episodes} aborted early due to SUMO crash. Resetting...")
            
        try:
            traci.close()
        except Exception:
            pass

        if mappo.n_episodes >= CONFIG['EPISODES_BEFORE_TRAIN']:
            mappo.train()
            
        if mappo.episode_done and ((mappo.n_episodes) % CONFIG['EVAL_INTERVAL'] == 0):
            trailing_avg = np.mean(mappo.episode_rewards[-CONFIG['EVAL_INTERVAL']:])
            print(f"Episode {mappo.n_episodes} | Avg Reward: {trailing_avg:.2f} | Demand: {train_demand} vph/lane | PR: {train_pr:.2f}")
            eval_rewards.append(trailing_avg)
            mappo.save(dirs['models'], mappo.n_episodes)

    mappo.save(dirs['models'], CONFIG['MAX_EPISODES'])

    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel(f"Evaluation Intervals (x{CONFIG['EVAL_INTERVAL']})")
    plt.ylabel("Average Stabilizing Reward")
    plt.title(f"MAPPO Curriculum Performance (Seed {args.seed})")
    plt.savefig(os.path.join(output_dir, "training_curve.png"))
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    train(args)