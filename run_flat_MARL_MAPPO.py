import os
import argparse
import random
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import traci

from MAPPO import MAPPO
from common.utils import init_dir

# --- DONGCHEN06 CONFIGURATIONS ---
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
    """Locks all random engines for absolute reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Train or Evaluate DongChen06 Flat-MARL baseline over SUMO')
    parser.add_argument('--base-dir', type=str, default="./results_dongchen_baseline/", help="experiment base dir")
    parser.add_argument('--option', type=str, default='train', help="train or evaluate")
    parser.add_argument('--model-dir', type=str, default='', help="pretrained model path")
    parser.add_argument('--gui', action='store_true', help="Run SUMO with GUI (sumo-gui)")
    
    parser.add_argument('--start-episode', type=int, default=0, help="Episode to resume training from")
    
    # Paper Reproducibility Args
    parser.add_argument('--seed', type=int, default=42, help="Global random seed")
    parser.add_argument('--eval-demand', type=int, default=800, help="Traffic lane demand for evaluation")
    parser.add_argument('--eval-pr', type=float, default=0.10, help="CAV Penetration Rate for evaluation")
    return parser.parse_args()


def generate_mixed_traffic_route(target_penetration_rate, lane_demand_vph, num_mainline_lanes=2):
    total_mainline_vph = lane_demand_vph * num_mainline_lanes
    ramp_vph = lane_demand_vph * 0.35 

    cav_main_vph = total_mainline_vph * target_penetration_rate
    hdv_main_vph = total_mainline_vph * (1.0 - target_penetration_rate)
    
    cav_ramp_vph = ramp_vph * target_penetration_rate
    hdv_ramp_vph = ramp_vph * (1.0 - target_penetration_rate)

    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <!-- HDV Behavior Distribution with physical emergency limits added -->
    <vTypeDistribution id="hdv_mixture">
        <vType id="hdv_cautious" probability="0.2" carFollowModel="IDM" accel="2.0" decel="4.5" emergencyDecel="7.5" tau="1.8" speedFactor="0.9" speedDev="0.05" lcCooperative="2.0" lcAssertive="0.1" color="0,255,0"/>
        <vType id="hdv_normal" probability="0.6" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" tau="1.2" speedFactor="1.0" speedDev="0.1" lcCooperative="1.0" lcAssertive="1.0" color="255,255,255"/>
        <vType id="hdv_aggressive" probability="0.2" carFollowModel="IDM" accel="3.5" decel="4.5" emergencyDecel="9.0" tau="0.8" speedFactor="1.15" speedDev="0.1" lcCooperative="0.1" lcAssertive="2.0" color="0,0,255"/>
    </vTypeDistribution>

    <!-- CAV Definition with emergency limit added -->
    <vType id="cav" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" tau="1.0" color="255,0,0"/>

    <route id="route_main" edges="E0 E1 E6" />
    <route id="route_ramp" edges="E_ramp E1 E6" />

    <flow id="human_flow" type="hdv_mixture" route="route_main" begin="0" end="10000" vehsPerHour="{hdv_main_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>
    <flow id="cav_main_flow" type="cav" route="route_main" begin="0" end="10000" vehsPerHour="{cav_main_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>
    
    <flow id="human_ramp_flow" type="hdv_mixture" route="route_ramp" begin="0" end="10000" vehsPerHour="{hdv_ramp_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>
    <flow id="cav_ramp_flow" type="cav" route="route_ramp" begin="0" end="10000" vehsPerHour="{cav_ramp_vph:.2f}" departLane="random" departSpeed="random" departPos="random"/>
</routes>
"""
    os.makedirs("SUMO_network", exist_ok=True)
    with open("SUMO_network/highway_onramp_actual.rou.xml", "w") as f:
        f.write(xml_content)


class DongChenFlatEnv:
    def __init__(self, num_cavs):
        self.num_cavs = num_cavs
        self.cav_prefix = "cav"
        self.state_dim = 6 
        self.action_dim = 5 

    def get_observations(self, dummy):
        obs_dict = {}
        try:
            active_cavs = [v for v in traci.vehicle.getIDList() if v.startswith(self.cav_prefix)]
        except traci.exceptions.TraCIException:
            return {}, []

        for cav in active_cavs:
            try:
                ego_speed = traci.vehicle.getSpeed(cav)
                ego_pos = traci.vehicle.getLanePosition(cav)

                leader_info = traci.vehicle.getLeader(cav, 150.0)
                if leader_info:
                    leader_speed = traci.vehicle.getSpeed(leader_info[0])
                    leader_dist = leader_info[1]
                else:
                    leader_speed = ego_speed
                    leader_dist = 150.0

                follower_info = traci.vehicle.getFollower(cav, 150.0)
                if follower_info and follower_info[0] != "":
                    follower_speed = traci.vehicle.getSpeed(follower_info[0])
                    follower_dist = follower_info[1]
                else:
                    follower_speed = ego_speed
                    follower_dist = 150.0
                
                obs_dict[cav] = [
                    ego_speed / 30.0, ego_pos / 1000.0, 
                    leader_speed / 30.0, leader_dist / 150.0,
                    follower_speed / 30.0, follower_dist / 150.0
                ]
            except traci.exceptions.TraCIException:
                pass
                
        return obs_dict, active_cavs

    def _compute_human_centric_reward(self, cav_id):
        try:
            follower = traci.vehicle.getFollower(cav_id, dist=100.0)
            collisions = traci.simulation.getCollidingVehiclesIDList()
            
            # --- FIX 3: CRASH OVERRIDE (Short-Circuit Penalty) ---
            # If the CAV crashes, or the human directly behind it crashes, 
            # instantly return a heavy penalty without adding new weights.
            if cav_id in collisions or (follower[0] != "" and follower[0] in collisions):
                return -10.0 
                
            w_eff, w_safe, w_stab, w_fair, w_ctrl = 1.0, 10.0, 5.0, 0.5, 0.1
            ego_speed = traci.vehicle.getSpeed(cav_id)
            ego_accel = traci.vehicle.getAcceleration(cav_id)
            
            r_eff, r_safe, r_stab, r_fair, c_ctrl = ego_speed / 20.0, 0.0, 0.0, 0.0, 0.0
            
            if follower[0] != "":
                follower_id = follower[0]
                follower_accel = traci.vehicle.getAcceleration(follower_id)
                follower_wait = traci.vehicle.getWaitingTime(follower_id)
                
                if follower_accel < -4.5:
                    r_safe -= 1.0
                if follower_accel < 0.0:
                    r_stab -= abs(follower_accel) / 5.0 
                if follower_wait > 0:
                    r_fair -= follower_wait / 100.0
                    
            if ego_accel < 0.0:
                c_ctrl += abs(ego_accel) / 5.0
                
            final_reward = (w_eff * r_eff) + (w_safe * r_safe) + (w_stab * r_stab) + (w_fair * r_fair) - (w_ctrl * c_ctrl)
            return float(final_reward)
        except traci.exceptions.TraCIException:
            return 0.0

    def step(self, action_dict):
        accel_map = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
        for cav, action_idx in action_dict.items():
            try:
                accel = accel_map[action_idx]
                
                # --- FIX 1: DISABLE TRACI SAFETY NET ---
                traci.vehicle.setSpeedMode(cav, 0)
                
                traci.vehicle.setAcceleration(cav, accel, duration=0.1)
            except traci.exceptions.TraCIException:
                pass
                
        traci.simulationStep()
        
        rewards = {}
        try:
            active_cavs = [v for v in traci.vehicle.getIDList() if v.startswith(self.cav_prefix)]
            for cav in active_cavs:
                rewards[cav] = self._compute_human_centric_reward(cav)
        except traci.exceptions.TraCIException:
            pass
            
        return rewards


class TraCIEnvWrapper:
    def __init__(self, num_cavs):
        self.env = DongChenFlatEnv(num_cavs=num_cavs)
        
    def reset(self):
        try:
            return self.env.get_observations({})
        except traci.exceptions.TraCIException:
            return {}, []
            
    def step(self, action_dict):
        return self.env.step(action_dict)
        
    def get_observations(self, dummy):
        return self.env.get_observations(dummy)


def train(args):
    set_global_seeds(args.seed)
    base_dir = args.base_dir
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = os.path.join(base_dir, "0809")
    dirs = init_dir(output_dir)

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual.rou.xml"
    NUM_CAVS = 5

    env_wrapper = TraCIEnvWrapper(num_cavs=NUM_CAVS)

    mappo = MAPPO(env=env_wrapper, memory_capacity=CONFIG['MEMORY_CAPACITY'],
                  state_dim=6, action_dim=5, num_cavs=NUM_CAVS,
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
        print("Beginning DongChen06 Baseline Curriculum Training from scratch across Multi-Densities...")

    if args.start_episode > 0:
        mappo.n_episodes = args.start_episode
        print(f"Manually setting start episode to {mappo.n_episodes}...")

    eval_rewards = []

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
        
        # --- FIX 2: ENABLE PHYSICAL COLLISIONS DURING TRAINING ---
        traci_args = [sumo_binary, "-n", net_file, "-r", route_file, "--seed", str(episode_seed), "--start", "--quit-on-end",
                      "--collision.action", "warn", "--collision.stoptime", "10", "--step-length", "0.1", "--no-step-log", "true", 
                      "--collision.check-junctions", "true", "--collision.mingap-factor", "0.0"]
        traci.start(traci_args)
        
        mappo.obs_dict, mappo.active_cavs = env_wrapper.get_observations({})
        mappo.n_steps = 0
        mappo.interact()
        traci.close()

        if mappo.n_episodes >= CONFIG['EPISODES_BEFORE_TRAIN']:
            mappo.train()
            
        if mappo.episode_done and ((mappo.n_episodes) % CONFIG['EVAL_INTERVAL'] == 0):
            trailing_avg = np.mean(mappo.episode_rewards[-CONFIG['EVAL_INTERVAL']:])
            print(f"Episode {mappo.n_episodes} | Avg DongChen Reward: {trailing_avg:.2f} | Demand: {train_demand} vph/lane | PR: {train_pr:.2f}")
            eval_rewards.append(trailing_avg)
            mappo.save(dirs['models'], mappo.n_episodes)

    mappo.save(dirs['models'], CONFIG['MAX_EPISODES'])

    plt.figure()
    plt.plot(eval_rewards, color='orange', linestyle='dashed', label="DongChen06 Baseline")
    plt.xlabel(f"Evaluation Intervals (x{CONFIG['EVAL_INTERVAL']})")
    plt.ylabel("Average Stabilizing Reward")
    plt.title("DongChen06 Flat-MARL Performance")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "dongchen_baseline_curve.png"))
    plt.show()


def evaluate(args):
    if not args.model_dir or not os.path.exists(args.model_dir):
        raise Exception("You must provide a valid --model-dir to evaluate a pre-trained model.")

    sumo_binary = "sumo-gui" if args.gui else "sumo"
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual.rou.xml"
    NUM_CAVS = 5

    print(f"Generating Evaluation Environment: Demand {args.eval_demand} vph/lane | PR {args.eval_pr*100}%")
    generate_mixed_traffic_route(target_penetration_rate=args.eval_pr, lane_demand_vph=args.eval_demand)

    env_wrapper = TraCIEnvWrapper(num_cavs=NUM_CAVS)
    mappo = MAPPO(env=env_wrapper, memory_capacity=CONFIG['MEMORY_CAPACITY'],
                  state_dim=6, action_dim=5, num_cavs=NUM_CAVS,
                  batch_size=CONFIG['BATCH_SIZE'], entropy_reg=CONFIG['ENTROPY_REG'],
                  roll_out_n_steps=CONFIG['ROLL_OUT_N_STEPS'],
                  actor_hidden_size=CONFIG['actor_hidden_size'], critic_hidden_size=CONFIG['critic_hidden_size'],
                  actor_lr=CONFIG['actor_lr'], critic_lr=CONFIG['critic_lr'], reward_scale=CONFIG['reward_scale'],
                  target_update_steps=CONFIG['TARGET_UPDATE_STEPS'], target_tau=CONFIG['TARGET_TAU'],
                  reward_gamma=CONFIG['reward_gamma'], reward_type=CONFIG['reward_type'],
                  max_grad_norm=CONFIG['MAX_GRAD_NORM'], episodes_before_train=CONFIG['EPISODES_BEFORE_TRAIN'])

    mappo.load(args.model_dir, train_mode=False)

    test_seeds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 325, 
                  350, 375, 400, 425, 450, 475, 500, 525, 550, 575]
    
    print(f"Beginning Evaluation on {len(test_seeds)} Seeds...")
    
    for ep, current_seed in enumerate(test_seeds):
        set_global_seeds(current_seed)
        
        # --- FIX 2: ENABLE PHYSICAL COLLISIONS DURING EVALUATION ---
        traci_args = [sumo_binary, "-n", net_file, "-r", route_file, "--seed", str(current_seed), "--start", "--quit-on-end",
                      "--collision.action", "warn", "--collision.stoptime", "10", 
                      "--collision.check-junctions", "true", "--collision.mingap-factor", "0.0"]
        traci.start(traci_args)
        
        obs_dict, active_cavs = env_wrapper.get_observations({})
        
        episode_reward = 0.0
        done = False
        step = 0
        
        hv_jerk_list = []
        hv_waiting_times = []
        hard_braking_events = 0
        dangerous_ttc_events = 0
        prev_accel = {}
        
        # --- FIX 4: COLLISION TRACKING SETS ---
        total_vehicles = set()
        collided_vehicles = set()
        
        while not done and step < 1000:
            step += 1
            
            try:
                all_vehicles = traci.vehicle.getIDList()
                collisions = traci.simulation.getCollidingVehiclesIDList()
                
                hvs = [v for v in all_vehicles if "human" in v]
                
                # Update collision metrics
                for v in all_vehicles:
                    total_vehicles.add(v)
                    if v in collisions:
                        collided_vehicles.add(v)
                
                for hv in hvs:
                    current_speed = traci.vehicle.getSpeed(hv)
                    current_accel = traci.vehicle.getAcceleration(hv)
                    
                    if hv in prev_accel:
                        jerk = abs(current_accel - prev_accel[hv])
                        hv_jerk_list.append(jerk)
                    prev_accel[hv] = current_accel
                    
                    if current_accel < -3.0: 
                        hard_braking_events += 1
                        
                    leader_info = traci.vehicle.getLeader(hv, 100.0)
                    if leader_info:
                        leader_id, distance = leader_info
                        leader_speed = traci.vehicle.getSpeed(leader_id)
                        relative_speed = current_speed - leader_speed
                        if relative_speed > 0:
                            ttc = distance / relative_speed
                            if ttc < 2.5:
                                dangerous_ttc_events += 1
                                
                    hv_waiting_times.append(traci.vehicle.getAccumulatedWaitingTime(hv))
            except traci.exceptions.TraCIException:
                pass
            
            state_array = np.zeros((NUM_CAVS, 6))
            cav_list = list(obs_dict.keys())
            
            for idx in range(min(NUM_CAVS, len(cav_list))):
                state_array[idx] = obs_dict[cav_list[idx]]
                
            action = mappo.action(state_array, NUM_CAVS)
            
            action_dict = {}
            for idx in range(min(NUM_CAVS, len(cav_list))):
                action_dict[cav_list[idx]] = action[idx]
                
            rewards_dict = env_wrapper.step(action_dict)
            obs_dict, current_active_cavs = env_wrapper.get_observations({})
            
            episode_reward += sum(rewards_dict.values())
            
            if len(current_active_cavs) == 0 and step > 50:
                done = True
                
        traci.close()
        
        avg_jerk = np.mean(hv_jerk_list) if hv_jerk_list else 0.0
        avg_wait = np.max(hv_waiting_times) if hv_waiting_times else 0.0 
        col_rate = (len(collided_vehicles) / max(1, len(total_vehicles))) * 100.0
        
        print(f"Eval Seed {current_seed} | Reward: {episode_reward:.2f} | "
              f"Collisions: {col_rate:.1f}% | Avg Jerk: {avg_jerk:.3f} m/s^3 | "
              f"Brakes(<-3m/s^2): {hard_braking_events} | TTCs(<2.5s): {dangerous_ttc_events} | "
              f"Max Wait: {avg_wait:.1f}s")


if __name__ == "__main__":
    args = parse_args()
    if args.option == 'train':
        train(args)
    elif args.option == 'evaluate':
        evaluate(args)