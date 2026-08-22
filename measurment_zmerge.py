import os
import argparse
import numpy as np
import traci

# Import your actual Z-Merge Agent and Environment
from z_merge_agent import ZMergeAgent
from z_merge_env import ZMergeEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained Z-Merge policy on SUMO')
    parser.add_argument('--model-prefix', type=str, required=True, help="Prefix of the saved model (e.g., Jul_28_03_52_04/zmerge_v1)")
    parser.add_argument('--gui', action='store_true', help="Run SUMO with GUI (sumo-gui)")
    parser.add_argument('--episodes', type=int, default=5, help="Number of evaluation episodes to run")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
        
    # SUMO Configuration
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual.rou.xml"
    
    # We now know this is exactly 22!
    state_dim = 22 
    RAMP_EDGE_ID = "E_ramp" 

    # Initialize the CORRECT environment
    print("Initializing ZMergeEnv...")
    env = ZMergeEnv(net_file, route_file, use_gui=args.gui)

    print(f"Loading Z-Merge Agent with prefix '{args.model_prefix}'...")
    agent = ZMergeAgent(state_dim=state_dim)
    try:
        agent.load(args.model_prefix)
    except FileNotFoundError:
        print(f"Error: Could not find model files matching prefix '{args.model_prefix}' in the results_zmerge folder.")
        return

    print(f"Beginning Evaluation for {args.episodes} episodes with Z-Merge Metrics Tracking...")
    
    # Global Trackers across all evaluation episodes
    all_efficiency_speeds = []
    all_comfort_accelerations = []
    all_queue_counts = []
    global_total_avs = set()
    global_collided_avs = set()
    global_ramp_tracked = set()
    global_ramp_stopped = set()
    global_ramp_collided = set()
    
    for ep in range(args.episodes):
        # ZMergeEnv's reset handles traci.start() for us automatically!
        obs_dict = env.reset()
        
        episode_reward = 0.0
        step = 0
        
        while step < 1000:
            step += 1
            cav_list = list(obs_dict.keys())
            action_dict = {}
            
            # Only ask the agent for actions if there are actually CAVs on the road
            if len(cav_list) > 0:
                for cav_id in cav_list:
                    state = obs_dict[cav_id]
                    # Get discrete action and continuous params (Epsilon 0.0 for pure evaluation)
                    d_action, c_params = agent.act(state, epsilon=0.0)
                    action_dict[cav_id] = (d_action, c_params)
                
            # Advance simulation using ZMergeEnv's native step function
            next_obs, rewards, dones = env.step(action_dict)
            obs_dict = next_obs
            episode_reward += sum(rewards.values()) if rewards else 0.0
            
            # --- START Z-MERGE METRICS TRACKING ---
            step_queue_count = 0
            colliding_vehicles = traci.simulation.getCollidingVehiclesIDList()
            all_vehicles = traci.vehicle.getIDList()
            
            for veh_id in all_vehicles:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    accel = traci.vehicle.getAcceleration(veh_id)
                    edge = traci.vehicle.getRoadID(veh_id)
                    
                    # Safety Tracking
                    is_av = veh_id.startswith("cav")
                    if is_av:
                        global_total_avs.add(veh_id + f"_ep{ep}") 
                        if veh_id in colliding_vehicles:
                            global_collided_avs.add(veh_id + f"_ep{ep}")
                    
                    # Efficiency & Comfort
                    all_efficiency_speeds.append(speed)
                    all_comfort_accelerations.append(accel)
                    
                    # Queue Tracking
                    if speed < 2.0:
                        step_queue_count += 1
                        
                    # Success Rate Tracking (Ramp Vehicles)
                    if edge == RAMP_EDGE_ID:
                        unique_ramp_id = veh_id + f"_ep{ep}"
                        global_ramp_tracked.add(unique_ramp_id)
                        if speed < 0.1: 
                            global_ramp_stopped.add(unique_ramp_id)
                        if veh_id in colliding_vehicles:
                            global_ramp_collided.add(unique_ramp_id)
                
                except traci.exceptions.TraCIException:
                    pass 
                        
            all_queue_counts.append(step_queue_count)
            # --- END Z-MERGE METRICS TRACKING ---

            # Safely end the episode early ONLY IF cars have spawned and then all left the map
            if step > 50 and len(all_vehicles) == 0:
                break
                
        print(f"Evaluation Episode {ep+1}/{args.episodes} | Total Reward: {episode_reward:.2f}")

    # Make sure to close TraCI after all episodes are finished
    traci.close()

    # --- CALCULATE & PRINT FINAL Z-MERGE METRICS ---
    avg_efficiency_speed = np.mean(all_efficiency_speeds) if all_efficiency_speeds else 0.0
    safety_collision_rate = (len(global_collided_avs) / len(global_total_avs)) * 100 if global_total_avs else 0.0
    comfort_variance = np.var(all_comfort_accelerations) if all_comfort_accelerations else 0.0
    avg_queue_length = np.mean(all_queue_counts) if all_queue_counts else 0.0
    
    failed_merges = global_ramp_stopped.union(global_ramp_collided)
    successful_merges = len(global_ramp_tracked) - len(failed_merges)
    success_rate = (successful_merges / len(global_ramp_tracked)) * 100 if global_ramp_tracked else 0.0

    report = (
        f"\n==================================================\n"
        f"FINAL Z-MERGE METHODOLOGY EVALUATION RESULTS\n"
        f"==================================================\n"
        f"Traffic Efficiency (Avg Speed): {avg_efficiency_speed:.2f} m/s\n"
        f"Traffic Safety (AV Collision Rate): {safety_collision_rate:.2f}%\n"
        f"Driving Comfort (Accel Variance): {comfort_variance:.2f} m/s^2\n"
        f"Average Queue Length: {avg_queue_length:.2f} vehicles/step\n"
        f"Merging Success Rate: {success_rate:.2f}%\n"
        f"==================================================\n"
    )

    print(report)

    # Clean up the output filename so it doesn't try to create extra folders
    safe_prefix = args.model_prefix.replace("/", "_").replace("\\", "_")
    output_filepath = f"results_zmerge/{safe_prefix}_evaluation_results.txt"
    
    with open(output_filepath, "w") as file:
        file.write(report)
        
    print(f"Successfully saved evaluation metrics to: {output_filepath}")

if __name__ == "__main__":
    main()