import os
import argparse
import numpy as np
import traci
import cv2
import shutil
import time
import warnings
import sys

# Hide the PyTorch log_softmax deprecation warning to keep logs clean
warnings.filterwarnings("ignore", category=UserWarning)

from MAPPO import MAPPO

def parse_args():
    parser = argparse.ArgumentParser(description='Record MP4 Videos of DongChen06 Baseline for all scenarios')
    parser.add_argument('--model-dir', type=str, required=True, help="Path to your saved model directory")
    parser.add_argument('--demand', type=str, default="800,1700", help="Comma-separated list of lane demands (veh/h/lane)")
    parser.add_argument('--pr', type=str, default="0.05,0.10,0.20", help="Comma-separated list of PRs")
    parser.add_argument('--seed', default=[0, 25, 50, 75, 100, 125, 150, 175, 200, 325, 350, 
                                           375, 400, 425, 450, 475, 500, 525, 550, 575], help="Random seeds to record")
    args = parser.parse_args()
    return args

def generate_mixed_traffic_route(target_penetration_rate, lane_demand_vph, num_mainline_lanes=2):
    """Generates the dynamic XML route file with mixed human driver behaviors using native vehsPerHour."""
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
        <vType id="hdv_aggressive" probability="0.2" carFollowModel="IDM" accel="3.5" decel="4.5" emergencyDecel="9.0" apparentDecel="4.5" tau="1.0" speedFactor="1.25" speedDev="0.1" lcCooperative="0.0" lcAssertive="5.0" color="0,0,255"/>
    </vTypeDistribution>
    <vType id="cav" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" tau="1.0" color="255,0,0"/>
    <route id="route_main" edges="E0 E1 E6" />
    <route id="route_ramp" edges="E_ramp E1 E6" />
    {flows_xml}
</routes>
"""
    os.makedirs("SUMO_network", exist_ok=True)
    with open("SUMO_network/highway_onramp_actual.rou.xml", "w") as f:
        f.write(xml_content)

def create_native_snapshot_settings(frame_dir, max_steps=4000, step_length=0.1, gui_settings_path="SUMO_network/gui_settings.xml"):
    """
    Configures SUMO to natively capture screenshots at C++ level every 2 steps.
    This bypasses TraCI screenshot IPC calls completely, preventing GUI thread deadlock.
    """
    # Removed the duplicate line
    abs_frame_dir = os.path.abspath(frame_dir).replace("\\", "/")
    
    # 1. Generate a list of explicit snapshot tags
    snapshots = []
    
    # Capture a frame every 2 steps (0.2 seconds)
    for step in range(2, max_steps, 2):
        sim_time = round(step * step_length, 2)
        # Generates frame_0002.png, frame_0004.png, etc.
        file_name = f"{abs_frame_dir}/frame_{step:04d}.png"
        
        # Use the 'time' attribute instead of 'period'
        snapshots.append(f'<snapshot time="{sim_time:.2f}" file="{file_name}"/>')
        
    # 2. Join the list into a single formatted string
    snapshots_xml = "\n    ".join(snapshots)
    
    # 3. Inject the unrolled tags into your XML content
    xml_content = f"""<viewsettings>
    <viewport zoom="250" x="600" y="0"/>
    <delay value="25"/>
    {snapshots_xml}
</viewsettings>"""

    # 4. Save the generated XML to the file (This was missing!)
    os.makedirs(os.path.dirname(gui_settings_path), exist_ok=True)
    with open(gui_settings_path, "w") as f:
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
        except Exception:
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
                    leader_speed, leader_dist = ego_speed, 150.0
                follower_info = traci.vehicle.getFollower(cav, 150.0)
                if follower_info and follower_info[0] != "":
                    follower_speed = traci.vehicle.getSpeed(follower_info[0])
                    follower_dist = follower_info[1]
                else:
                    follower_speed, follower_dist = ego_speed, 150.0
                
                obs_dict[cav] = [ego_speed / 30.0, ego_pos / 1000.0, leader_speed / 30.0, leader_dist / 150.0, follower_speed / 30.0, follower_dist / 150.0]
            except Exception:
                pass
        return obs_dict, active_cavs

    def _compute_human_centric_reward(self, cav_id):
        try:
            w_eff, w_safe, w_stab, w_fair, w_ctrl = 1.0, 10.0, 5.0, 0.5, 0.1
            ego_speed = traci.vehicle.getSpeed(cav_id)
            ego_accel = traci.vehicle.getAcceleration(cav_id)
            follower = traci.vehicle.getFollower(cav_id, dist=100.0)
            
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
        except Exception:
            return 0.0

    def step(self, action_dict):
        accel_map = {0: -3.0, 1: -1.5, 2: 0.0, 3: 1.5, 4: 3.0}
        for cav, action_idx in action_dict.items():
            try: 
                traci.vehicle.setSpeedMode(cav, 0)
                traci.vehicle.setAcceleration(cav, accel_map[action_idx], duration=0.1)
            except Exception: pass
            
        try:
            traci.simulationStep()
        except Exception:
            pass
            
        try:
            collided_vehs = set(traci.simulation.getCollidingVehiclesIDList())
        except Exception:
            collided_vehs = set()
            
        try:
            active_cavs = set([v for v in traci.vehicle.getIDList() if v.startswith(self.cav_prefix)])
        except Exception:
            active_cavs = set()
            
        rewards = {}
        for cav_id in action_dict.keys():
            if cav_id in collided_vehs:
                rewards[cav_id] = -10.0  # Terminal crash penalty
            elif cav_id in active_cavs:
                rewards[cav_id] = self._compute_human_centric_reward(cav_id)
            else:
                rewards[cav_id] = 0.0
                
        return rewards

class TraCIEnvWrapper:
    def __init__(self, num_cavs):
        self.env = DongChenFlatEnv(num_cavs=num_cavs)
        
    def reset(self):
        try: return self.env.get_observations({})
        except Exception: return {}, []
        
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
            
        return {cav: rewards.get(cav, 0.0) for cav in action_dict.keys()}
        
    def get_observations(self, dummy): 
        try:
            obs_dict, active_cavs = self.env.get_observations(dummy)
            active_vehs = set(traci.vehicle.getIDList())
            safe_obs = {cav: obs for cav, obs in obs_dict.items() if cav in active_vehs}
            safe_active = [cav for cav in active_cavs if cav in active_vehs]
            return safe_obs, safe_active
        except Exception:
            return {}, []

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_dir):
        raise Exception(f"Sorry, no pretrained model found at: {args.model_dir}")
        
    save_dir = os.path.dirname(args.model_dir) if os.path.isfile(args.model_dir) else args.model_dir
    recordings_dir = os.path.join(save_dir, "recordings")
    if not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
        
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual.rou.xml"
    gui_settings_file = "SUMO_network/gui_settings.xml"
    
    NUM_CAVS = 5
    state_dim = 6
    action_dim = 5
    MAX_STEPS = 4000  # 400 seconds horizon at 0.1s step length

    demands = [int(d) for d in args.demand.split(',')]
    prs = [float(pr) for pr in args.pr.split(',')]
    test_seeds = args.seed if isinstance(args.seed, list) else [int(s) for s in str(args.seed).split(',')]

    print(f"Loading MAPPO Agent from {args.model_dir}...", flush=True)
    env_wrapper = TraCIEnvWrapper(num_cavs=NUM_CAVS)
    mappo = MAPPO(env=env_wrapper, state_dim=state_dim, action_dim=action_dim, num_cavs=NUM_CAVS, actor_hidden_size=256, critic_hidden_size=256)
    mappo.load(args.model_dir, train_mode=False)

    for demand in demands:
        for pr in prs:
            print(f"\n==================================================")
            print(f"PREPARING ROUTE: Lane Demand {demand} vph | PR {pr*100:.0f}%")
            print(f"==================================================")
            
            generate_mixed_traffic_route(target_penetration_rate=pr, lane_demand_vph=demand)

            for current_seed in test_seeds:
                print(f"--> RECORDING VIDEO: Demand {demand} | PR {pr*100:.0f}% | Seed {current_seed}", flush=True)
                
                frame_dir = f"temp_frames_DONGCHEN_d{demand}_p{int(pr*100)}_s{current_seed}"
                if os.path.exists(frame_dir):
                    shutil.rmtree(frame_dir)
                os.makedirs(frame_dir)

                # Set up native snapshot export in XML
                create_native_snapshot_settings(frame_dir, max_steps=MAX_STEPS, step_length=0.1, gui_settings_path=gui_settings_file)

                traci_args = [
                    "sumo-gui", "-n", net_file, "-r", route_file,
                    "-g", gui_settings_file,
                    "--seed", str(current_seed), "--start", "--quit-on-end", 
                    "--step-length", "0.1",
                    "--no-step-log", "true", "--no-warnings", "true", 
                    "--window-size", "1280,720", "--window-pos", "50,50",
                    "--collision.action", "remove",
                    "--collision.check-junctions", "true", 
                    "--collision.mingap-factor", "0.0"
                ]
                              
                traci.start(traci_args)
                
                obs_dict, active_cavs = env_wrapper.get_observations({})
                step = 0
                episode_reward = 0.0
                
                # Run Simulation Steps (Pure RL execution loop)
                try:
                    while step < MAX_STEPS:
                        step += 1
                        state_array = np.zeros((NUM_CAVS, state_dim))
                        cav_list = list(obs_dict.keys())
                        action_dict = {}
                        
                        if len(cav_list) > 0:
                            for idx in range(min(NUM_CAVS, len(cav_list))):
                                state_array[idx] = obs_dict[cav_list[idx]]
                                
                            action = mappo.action(state_array, NUM_CAVS)
                            for idx in range(min(NUM_CAVS, len(cav_list))):
                                action_dict[cav_list[idx]] = action[idx]
                                
                        rewards = env_wrapper.step(action_dict)
                        episode_reward += sum(rewards.values())
                        obs_dict, current_active_cavs = env_wrapper.get_observations({})
                        
                        time.sleep(0.05)
                except Exception:
                    print(f"  [Warning] Episode aborted early due to SUMO error. Recovering...")
                        
                try:
                    traci.close()
                except Exception:
                    pass
                
                # Encode MP4 from natively exported snapshots
                print(f"    Simulation complete for Seed {current_seed} | Reward: {episode_reward:.2f}. Encoding MP4 video...", flush=True)
                images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
                images.sort()
                
                if images:
                    first_frame = cv2.imread(os.path.join(frame_dir, images[0]))
                    if first_frame is not None:
                        height, width, _ = first_frame.shape
                        video_name = os.path.join(recordings_dir, f'DongChen_LaneD{demand}_PR{int(pr*100)}_Seed{current_seed}.mp4')
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                        video = cv2.VideoWriter(video_name, fourcc, 10.0, (width, height))
                        
                        for image in images:
                            frame = cv2.imread(os.path.join(frame_dir, image))
                            if frame is not None:
                                video.write(frame)
                        video.release()
                        print(f"    Success! Video saved to: {video_name}")
                else:
                    print(f"    Warning: No screenshot frames captured for seed {current_seed}.")
                    
                # Clean up temporary frames directory
                if os.path.exists(frame_dir):
                    shutil.rmtree(frame_dir)

if __name__ == "__main__":
    main()