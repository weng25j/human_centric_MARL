import os
import argparse
import numpy as np
import traci
import cv2
import shutil
import time
import warnings
import matplotlib.pyplot as plt

# Hide the PyTorch log_softmax deprecation warning to keep eval logs clean
warnings.filterwarnings("ignore", category=UserWarning)

from MAPPO import MAPPO
from traffic_shepherd import TrafficShepherdEnv

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a trained MAPPO policy on SUMO')
    parser.add_argument('--model-dir', type=str, required=True, help="Path to saved model directory or .pt file")
    parser.add_argument('--gui', action='store_true', help="Force GUI for ALL episodes")
    parser.add_argument('--record-video', action='store_true', help="Record MP4 videos for each evaluated seed")
    parser.add_argument('--eval-lane-demands', type=str, default="800,1700", help="Comma-separated list of lane demands")
    parser.add_argument('--eval-prs', type=str, default="0.05,0.10,0.20", help="Comma-separated list of PRs")
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
        <vType id="hdv_normal" probability="0.4" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="8.0" apparentDecel="4.5" tau="1.2" speedFactor="1.0" speedDev="0.1" lcCooperative="1.0" lcAssertive="1.0" color="255,255,255"/>
        <vType id="hdv_aggressive" probability="0.4" carFollowModel="IDM" accel="3.5" decel="4.5" emergencyDecel="9.0" apparentDecel="4.5" tau="0.8" speedFactor="1.15" speedDev="0.1" lcCooperative="0.1" lcAssertive="2.0" color="0,0,255"/>
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

def create_native_snapshot_settings(frame_dir, gui_settings_path="SUMO_network/gui_settings.xml"):
    abs_frame_dir = os.path.abspath(frame_dir).replace("\\", "/")
    xml_content = f"""<viewsettings>
    <viewport zoom="800" x="500" y="0"/>
    <delay value="0"/>
    <snapshot period="0.2" file="{abs_frame_dir}/frame_%04d.png"/>
</viewsettings>"""
    os.makedirs(os.path.dirname(gui_settings_path), exist_ok=True)
    with open(gui_settings_path, "w") as f:
        f.write(xml_content)

class TraCIEnvWrapper:
    def __init__(self, net_file, route_file, num_cavs):
        self.env = TrafficShepherdEnv(net_file, route_file, num_cavs=num_cavs)
        
    def reset(self):
        try:
            return self.env.get_observations({})
        except traci.exceptions.TraCIException:
            return {}, []
            
    def step(self, action_dict):
        return self.env.step(action_dict)
        
    def get_observations(self, dummy):
        return self.env.get_observations(dummy)

def calc_mean_ci(data):
    if not data or len(data) == 0:
        return 0.0, 0.0
    mean = np.mean(data)
    ci = 1.96 * np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0.0
    return mean, ci

def main():
    args = parse_args()
    
    if not os.path.exists(args.model_dir):
        raise Exception(f"Sorry, no pretrained model found at: {args.model_dir}")
        
    save_dir = os.path.dirname(args.model_dir) if os.path.isfile(args.model_dir) else args.model_dir
    recordings_dir = os.path.join(save_dir, "recordings")
    if args.record_video and not os.path.exists(recordings_dir):
        os.makedirs(recordings_dir)
        
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual_shepherd.rou.xml"
    gui_settings_file = "SUMO_network/gui_settings.xml"
    
    NUM_CAVS = 5
    state_dim = 14  
    action_dim = 3  
    
    step_length = 0.1 
    TTC_THRESHOLD = 2.5 
    HARD_BRAKE_THRESHOLD = -3.0 
    NEAR_COLLISION_TTC_THRESHOLD = 1.5 
    MAX_STEPS = 4000 

    test_seeds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 325, 
                  350, 375, 400, 425, 450, 475, 500, 525, 550, 575]

    lane_demands = [int(d) for d in args.eval_lane_demands.split(',')]
    prs = [float(pr) for pr in args.eval_prs.split(',')]

    print(f"Loading MAPPO Agent from {args.model_dir}...", flush=True)
    env_wrapper = TraCIEnvWrapper(net_file, route_file, num_cavs=NUM_CAVS)
    mappo = MAPPO(env=env_wrapper, state_dim=state_dim, action_dim=action_dim, num_cavs=NUM_CAVS, actor_hidden_size=256, critic_hidden_size=256)
    mappo.load(args.model_dir, train_mode=False)

    for lane_demand in lane_demands:
        for pr in prs:
            print(f"\n==================================================")
            print(f"EVALUATING SHEPHERD: Lane Demand {lane_demand} veh/h/lane | PR {pr*100:.0f}%")
            print(f"==================================================")
            
            generate_mixed_traffic_route(target_penetration_rate=pr, lane_demand_vph=lane_demand)
            
            agg_results = {
                'hourly_throughput': [], 'raw_arrivals': [], 'avg_speed': [], 'avg_tt': [],
                'collision_rate': [], 'near_collisions': [], 'ttc_violations': [], 'hard_braking': [],
                'accel_var': [], 'jerk': [], 'wave_intensity': [],
                'hdv_delay_mean': [], 'hdv_delay_90th': [], 'success_rate': [],
                'avg_max_queue': [], 'abs_max_queue': [], 
                'interventions_total': [], 'interventions_per_cav': [], 'intervention_duty_cycle': [],
                'mean_decel': [], 'mean_duration': []
            }
            
            # Arrays for tracking metrics over time
            all_seeds_speed_over_time = np.zeros((len(test_seeds), MAX_STEPS))
            all_seeds_decel_over_time = np.zeros((len(test_seeds), MAX_STEPS))
            all_seeds_jerk_over_time = np.zeros((len(test_seeds), MAX_STEPS))
            
            all_raw_decels = []
            all_raw_inference_times = []

            for seed_idx, current_seed in enumerate(test_seeds):
                current_binary = "sumo-gui" if (args.gui or args.record_video) else "sumo"
                frame_dir = f"temp_frames_SHEPHERD_ld{lane_demand}_p{int(pr*100)}_s{current_seed}"
                
                traci_args = [current_binary, "-n", net_file, "-r", route_file, 
                              "--seed", str(current_seed), "--start", "--quit-on-end", 
                              "--step-length", str(step_length),
                              "--no-step-log", "true", "--no-warnings", "true", 
                              "--window-size", "1280,720", "--window-pos", "50,50",
                              "--collision.action", "remove",
                              "--collision.check-junctions", "true", 
                              "--collision.mingap-factor", "0.0"]
                
                if current_binary == "sumo-gui":
                    traci_args.extend(["--delay", "50"])
                
                if args.record_video:
                    if os.path.exists(frame_dir):
                        shutil.rmtree(frame_dir)
                    os.makedirs(frame_dir)
                    create_native_snapshot_settings(frame_dir, gui_settings_file)
                    traci_args.extend(["-g", gui_settings_file])
                              
                traci.start(traci_args)
                
                seed_metrics = {
                    'throughput': 0, 'travel_times': [], 'speeds': [],
                    'total_vehs': set(), 'total_cavs': set(), 'collided_vehs': set(), 
                    'ttc_violations': 0, 'near_collisions': 0,
                    'hard_braking_events': 0, 'accels': [], 'jerks': [],
                    'speed_variances': [], 'hdv_time_losses': [],
                    'ramp_vehs_total': set(), 'ramp_vehs_failed': set(),
                    'ramp_vehs_merged': set(), 'step_max_queues': [],
                    'cav_interventions': 0, 'cav_active_steps': {},
                    'cav_decel_magnitudes': [], 'cav_decel_durations': [],
                    'inference_times': []
                }
                
                obs_dict, active_cavs = env_wrapper.get_observations({})
                veh_track = {}
                completed_veh_track = {}
                active_cav_braking = {}
                active_ttc_events = set()
                active_near_col_events = set()
                active_hard_brakes = set()
                
                episode_reward = 0.0
                step = 0
                                                                    
                while step < MAX_STEPS:
                    step += 1
                    state_array = np.zeros((NUM_CAVS, state_dim))
                    cav_list = list(obs_dict.keys())
                    action_dict = {}
                    
                    current_step_speeds = []
                    current_step_decels = []
                    current_step_jerks = []
                    
                    if len(cav_list) > 0:
                        for idx in range(min(NUM_CAVS, len(cav_list))):
                            state_array[idx] = obs_dict[cav_list[idx]]
                            
                        # --- INFERENCE TIME TRACKING ---
                        inf_start = time.perf_counter()
                        action = mappo.action(state_array, NUM_CAVS)
                        inf_end = time.perf_counter()
                        
                        inf_time_ms = (inf_end - inf_start) * 1000.0
                        seed_metrics['inference_times'].append(inf_time_ms)
                        # -------------------------------
                        
                        for idx in range(min(NUM_CAVS, len(cav_list))):
                            action_dict[cav_list[idx]] = action[idx]
                            
                    spatial_rewards_dict = env_wrapper.step(action_dict)
                    obs_dict, current_active_cavs = env_wrapper.get_observations({})
                    episode_reward += sum(spatial_rewards_dict.values())
                    
                    # Catch collisions immediately after the step
                    collisions = traci.simulation.getCollidingVehiclesIDList()
                    for c_veh in collisions:
                        c_unique = f"{c_veh}_s{current_seed}"
                        seed_metrics['collided_vehs'].add(c_unique)
                        seed_metrics['total_vehs'].add(c_unique)
                        if c_veh.startswith("cav"):
                            seed_metrics['total_cavs'].add(c_unique)
                            
                        if c_veh in veh_track and veh_track[c_veh]['route'] == "route_ramp":
                            seed_metrics['ramp_vehs_total'].add(c_unique)
                            seed_metrics['ramp_vehs_failed'].add(c_unique)

                    if args.record_video and step > 1:
                        if step % 2 == 0:
                            pass
                    
                    all_vehicles = traci.vehicle.getIDList()
                    arrived_vehs = traci.simulation.getArrivedIDList()
                    
                    seed_metrics['throughput'] += len(arrived_vehs)
                    for arr_v in arrived_vehs:
                        arr_unique = f"{arr_v}_s{current_seed}"
                        if arr_unique in seed_metrics['ramp_vehs_total']:
                            seed_metrics['ramp_vehs_merged'].add(arr_unique)
                        if arr_v in active_cav_braking:
                            seed_metrics['cav_decel_durations'].append(active_cav_braking[arr_v] * step_length)
                            del active_cav_braking[arr_v]
                        if arr_v in veh_track:
                            seed_metrics['travel_times'].append((step - veh_track[arr_v]['entry_step']) * step_length)
                            if not veh_track[arr_v]['is_cav']:
                                seed_metrics['hdv_time_losses'].append(veh_track[arr_v]['max_time_loss'])
                            completed_veh_track[arr_v] = veh_track[arr_v]  # Retain for profile plotting
                            del veh_track[arr_v]
                    
                    for veh_id in all_vehicles:
                        try:
                            speed = traci.vehicle.getSpeed(veh_id)
                            current_step_speeds.append(speed)
                            seed_metrics['speeds'].append(speed)  # FIX: Populate main speeds array
                            accel = traci.vehicle.getAcceleration(veh_id)
                            
                            if accel < 0:
                                current_step_decels.append(abs(accel))
                                
                            time_loss = traci.vehicle.getTimeLoss(veh_id)
                            is_cav = veh_id.startswith("cav")
                            unique_id = f"{veh_id}_s{current_seed}"
                            route_id = traci.vehicle.getRouteID(veh_id)
                            
                            if is_cav:
                                seed_metrics['total_cavs'].add(unique_id)
                                seed_metrics['cav_active_steps'][unique_id] = seed_metrics['cav_active_steps'].get(unique_id, 0) + 1
                                if accel < -0.1:  
                                    seed_metrics['cav_interventions'] += 1
                                    seed_metrics['cav_decel_magnitudes'].append(abs(accel))
                                    active_cav_braking[veh_id] = active_cav_braking.get(veh_id, 0) + 1
                                else:
                                    if veh_id in active_cav_braking:
                                        seed_metrics['cav_decel_durations'].append(active_cav_braking[veh_id] * step_length)
                                        del active_cav_braking[veh_id]
                            
                            if route_id == "route_ramp":
                                seed_metrics['ramp_vehs_total'].add(unique_id)
                                if speed < 0.1 or veh_id in collisions:
                                    seed_metrics['ramp_vehs_failed'].add(unique_id)
                            
                            if veh_id not in veh_track:
                                veh_track[veh_id] = {
                                    'entry_step': step, 'prev_accel': accel,
                                    'max_time_loss': 0.0, 'is_cav': is_cav,
                                    'route': route_id, 'speed_history': [], 
                                    'accel_history': [], 'pos_history': []
                                }

                            try:
                                current_edge = traci.vehicle.getRoadID(veh_id)
                                if current_edge == "E1":
                                    pos_m = traci.vehicle.getLanePosition(veh_id)
                                else:
                                    pos_m = -1.0
                                veh_track[veh_id]['pos_history'].append(pos_m)
                            except:
                                veh_track[veh_id]['pos_history'].append(-1.0)
                                
                            veh_track[veh_id]['accel_history'].append(accel)

                            seed_metrics['accels'].append(accel)
                            prev_accel = veh_track[veh_id]['prev_accel']
                            jerk = (accel - prev_accel) / step_length
                            seed_metrics['jerks'].append(jerk)
                            current_step_jerks.append(abs(jerk))
                            
                            veh_track[veh_id]['prev_accel'] = accel
                            veh_track[veh_id]['speed_history'].append(speed)
                            seed_metrics['total_vehs'].add(unique_id)
                            
                            if veh_id in collisions:
                                seed_metrics['collided_vehs'].add(unique_id)
                                
                            if accel < HARD_BRAKE_THRESHOLD:
                                if veh_id not in active_hard_brakes:
                                    seed_metrics['hard_braking_events'] += 1
                                    active_hard_brakes.add(veh_id)
                            else:
                                active_hard_brakes.discard(veh_id)
                                
                            leader_info = traci.vehicle.getLeader(veh_id, 100.0)
                            if leader_info is not None:
                                leader_id, dist = leader_info
                                rel_speed = speed - traci.vehicle.getSpeed(leader_id)
                                if rel_speed > 0: 
                                    ttc = dist / rel_speed
                                    if ttc < TTC_THRESHOLD:
                                        if veh_id not in active_ttc_events:
                                            seed_metrics['ttc_violations'] += 1
                                            active_ttc_events.add(veh_id)
                                    else:
                                        active_ttc_events.discard(veh_id)

                                    if ttc < NEAR_COLLISION_TTC_THRESHOLD:
                                        if veh_id not in active_near_col_events:
                                            seed_metrics['near_collisions'] += 1
                                            active_near_col_events.add(veh_id)
                                    else:
                                        active_near_col_events.discard(veh_id)
                                else:
                                    active_ttc_events.discard(veh_id)
                                    active_near_col_events.discard(veh_id)
                            else:
                                active_ttc_events.discard(veh_id)
                                active_near_col_events.discard(veh_id)
                            
                            if time_loss > veh_track[veh_id]['max_time_loss']:
                                veh_track[veh_id]['max_time_loss'] = time_loss
                        except traci.exceptions.TraCIException:
                            pass 

                    step_max_queue = 0
                    for lane in traci.lane.getIDList():
                        lane_vehs = traci.lane.getLastStepVehicleIDs(lane)
                        current_queue = 0
                        for v in lane_vehs:
                            try:
                                if traci.vehicle.getSpeed(v) < 2.0:
                                    current_queue += 1
                                    step_max_queue = max(step_max_queue, current_queue)
                                else:
                                    current_queue = 0
                            except Exception:
                                pass
                    seed_metrics['step_max_queues'].append(step_max_queue)

                    # Update step arrays
                    if current_step_speeds:
                        all_seeds_speed_over_time[seed_idx, step - 1] = np.mean(current_step_speeds)
                    else:
                        all_seeds_speed_over_time[seed_idx, step - 1] = all_seeds_speed_over_time[seed_idx, step - 2] if step > 1 else 0.0
                        
                    if current_step_decels:
                        all_seeds_decel_over_time[seed_idx, step - 1] = np.mean(current_step_decels)
                    else:
                        all_seeds_decel_over_time[seed_idx, step - 1] = 0.0
                        
                    if current_step_jerks:
                        all_seeds_jerk_over_time[seed_idx, step - 1] = np.mean(current_step_jerks)
                    else:
                        all_seeds_jerk_over_time[seed_idx, step - 1] = 0.0

                for steps_b in active_cav_braking.values():
                    seed_metrics['cav_decel_durations'].append(steps_b * step_length)

                traci.close()
                print(f"  Seed {current_seed} Complete | Final Reward: {episode_reward:.2f}", flush=True)

                # =======================================================
                # DRIVING COMFORT PROFILES (AGGREGATED WORST-CASE)
                # =======================================================
                routes = ['route_main', 'route_ramp']
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                fig.subplots_adjust(wspace=0.3)
                
                # Combine currently active and arrived vehicles for plotting
                combined_veh_track = {**veh_track, **completed_veh_track}
                
                for idx, route in enumerate(routes):
                    ax = axes[idx]
                    candidates = []
                    for vid, data in combined_veh_track.items():
                        if data['route'] == route:
                            if any(0 <= p <= 300 for p in data['pos_history']):
                                candidates.append((vid, data))
                                    
                    if candidates:
                        # Defensive check added to lambda to prevent division by zero
                        worst_vid, worst_data = max(candidates, key=lambda x: np.var(x[1]['accel_history']) if len(x[1]['accel_history']) > 0 else 0.0)
                        plot_pos, plot_acc = [], []
                        for p, a in zip(worst_data['pos_history'], worst_data['accel_history']):
                            if 0 <= p <= 300: 
                                plot_pos.append(p)
                                plot_acc.append(a)
                        ax.plot(plot_pos, plot_acc, color='#1f77b4', linewidth=1.5, label='Traffic Shepherd')
                        
                    ax.axhline(1.47, color='green', linestyle='--', alpha=0.5, label='Comfortable (±1.47 m/s^2)')
                    ax.axhline(-1.47, color='green', linestyle='--', alpha=0.5)
                    ax.axhline(2.12, color='red', linestyle='--', alpha=0.5, label='Acceptable (±2.12 m/s^2)')
                    ax.axhline(-2.12, color='red', linestyle='--', alpha=0.5)
                    ax.set_xlim(0, 300)
                    ax.set_ylim(-4, 2.5)
                    ax.set_xlabel("Edge Position (m)")
                    ax.set_ylabel("Accel (m/s^2)")
                    title_name = "Mainline Route (E1)" if route == 'route_main' else "Ramp Route (E1)"
                    ax.set_title(title_name)
                    ax.legend(fontsize=9, loc='lower left')

                plt.suptitle(f"Worst-Case Driving Comfort Acceleration Profiles (Lane Demand {lane_demand} vph | PR {int(pr*100)}%) - Seed {current_seed}", fontsize=13)
                accel_plot_filename = os.path.join(save_dir, f"comfort_profiles_SHEPHERD_ld{lane_demand}_p{int(pr*100)}_s{current_seed}.png")
                plt.savefig(accel_plot_filename, bbox_inches='tight', dpi=300)
                plt.close()

                if args.record_video and os.path.exists(frame_dir):
                    images = [img for img in os.listdir(frame_dir) if img.endswith(".png")]
                    images.sort()
                    if images:
                        first_frame = cv2.imread(os.path.join(frame_dir, images[0]))
                        if first_frame is not None:
                            height, width, _ = first_frame.shape
                            video_name = os.path.join(recordings_dir, f'Shepherd_LaneD{lane_demand}_PR{int(pr*100)}_Seed{current_seed}.mp4')
                            video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (width, height))
                            for image in images:
                                frame = cv2.imread(os.path.join(frame_dir, image))
                                if frame is not None:
                                    video.write(frame)
                            video.release()
                    shutil.rmtree(frame_dir)

                # Store raw arrays for global CDF calculations
                all_raw_decels.extend(seed_metrics['cav_decel_magnitudes'])
                all_raw_inference_times.extend(seed_metrics['inference_times'])

                hourly_multiplier = 3600.0 / (MAX_STEPS * step_length)
                agg_results['hourly_throughput'].append(seed_metrics['throughput'] * hourly_multiplier)
                agg_results['raw_arrivals'].append(seed_metrics['throughput'])
                agg_results['avg_speed'].append(np.mean(seed_metrics['speeds']) if seed_metrics['speeds'] else 0.0)
                agg_results['avg_tt'].append(np.mean(seed_metrics['travel_times']) if seed_metrics['travel_times'] else 0.0)
                col_rate = (len(seed_metrics['collided_vehs']) / len(seed_metrics['total_vehs'])) * 100 if seed_metrics['total_vehs'] else 0.0
                agg_results['collision_rate'].append(col_rate)
                agg_results['near_collisions'].append(seed_metrics['near_collisions'])
                agg_results['ttc_violations'].append(seed_metrics['ttc_violations'])
                agg_results['hard_braking'].append(seed_metrics['hard_braking_events'])
                agg_results['accel_var'].append(np.var(seed_metrics['accels']) if seed_metrics['accels'] else 0.0)
                agg_results['jerk'].append(np.mean(np.abs(seed_metrics['jerks'])) if seed_metrics['jerks'] else 0.0)
                
                for v_data in list(veh_track.values()) + list(completed_veh_track.values()):
                    if len(v_data['speed_history']) > 10:
                        seed_metrics['speed_variances'].append(np.var(v_data['speed_history']))
                agg_results['wave_intensity'].append(np.mean(seed_metrics['speed_variances']) if seed_metrics['speed_variances'] else 0.0)
                
                agg_results['hdv_delay_mean'].append(np.mean(seed_metrics['hdv_time_losses']) if seed_metrics['hdv_time_losses'] else 0.0)
                agg_results['hdv_delay_90th'].append(np.percentile(seed_metrics['hdv_time_losses'], 90) if seed_metrics['hdv_time_losses'] else 0.0)
                
                successful_merges = len(seed_metrics['ramp_vehs_merged'] - seed_metrics['ramp_vehs_failed'])
                total_ramp = len(seed_metrics['ramp_vehs_total'])
                agg_results['success_rate'].append((successful_merges / total_ramp * 100) if total_ramp > 0 else 100.0)
                agg_results['avg_max_queue'].append(np.mean(seed_metrics['step_max_queues']) if seed_metrics['step_max_queues'] else 0.0)
                agg_results['abs_max_queue'].append(np.max(seed_metrics['step_max_queues']) if seed_metrics['step_max_queues'] else 0.0)
                
                num_cavs_in_seed = len(seed_metrics['total_cavs'])
                agg_results['interventions_total'].append(seed_metrics['cav_interventions'])
                agg_results['interventions_per_cav'].append(seed_metrics['cav_interventions'] / max(1, num_cavs_in_seed))
                agg_results['intervention_duty_cycle'].append((seed_metrics['cav_interventions'] / max(1, sum(seed_metrics['cav_active_steps'].values()))) * 100.0)
                agg_results['mean_decel'].append(np.mean(seed_metrics['cav_decel_magnitudes']) if seed_metrics['cav_decel_magnitudes'] else 0.0)
                agg_results['mean_duration'].append(np.mean(seed_metrics['cav_decel_durations']) if seed_metrics['cav_decel_durations'] else 0.0)

            # --- ARRAYS FOR CROSS-MODEL PLOTTING ---
            avg_speed_over_time = np.mean(all_seeds_speed_over_time, axis=0)
            avg_decel_over_time = np.mean(all_seeds_decel_over_time, axis=0)
            avg_jerk_over_time = np.mean(all_seeds_jerk_over_time, axis=0)
            
            speed_array_filename = os.path.join(save_dir, f"speed_array_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.npy")
            decel_array_filename = os.path.join(save_dir, f"decel_array_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.npy")
            jerk_array_filename = os.path.join(save_dir, f"jerk_array_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.npy")
            
            np.save(speed_array_filename, avg_speed_over_time)
            np.save(decel_array_filename, avg_decel_over_time)
            np.save(jerk_array_filename, avg_jerk_over_time)

            # --- PLOT: AVERAGE SPEED ---
            plt.figure(figsize=(8, 4))
            plt.plot(range(MAX_STEPS), avg_speed_over_time, label='Traffic Shepherd', color='#1f77b4') 
            plt.xlabel('Simulation Step', fontsize=12)
            plt.ylabel('Average Speed (m/s)', fontsize=12)
            plt.xlim([0, MAX_STEPS])
            ticks = np.arange(0, MAX_STEPS + 1, 1000)
            plt.xticks(ticks, [f"{int(t/1000)}k" if t > 0 else "0k" for t in ticks], fontsize=11)
            plt.yticks(fontsize=11)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='both', right=True, top=True)
            plt.legend(frameon=False, loc='upper center', ncol=4, fontsize=11)
            plt.savefig(os.path.join(save_dir, f"speed_plot_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.png"), bbox_inches='tight', dpi=300)
            plt.close()
            
            # --- PLOT: MEAN DECELERATION ---
            plt.figure(figsize=(8, 4))
            plt.plot(range(MAX_STEPS), avg_decel_over_time, label='Traffic Shepherd', color='#d62728') 
            plt.xlabel('Simulation Step', fontsize=12)
            plt.ylabel('Mean Decel Magnitude (m/s²)', fontsize=12)
            plt.xlim([0, MAX_STEPS])
            plt.xticks(ticks, [f"{int(t/1000)}k" if t > 0 else "0k" for t in ticks], fontsize=11)
            plt.yticks(fontsize=11)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='both', right=True, top=True)
            plt.legend(frameon=False, loc='upper center', ncol=4, fontsize=11)
            plt.savefig(os.path.join(save_dir, f"decel_plot_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.png"), bbox_inches='tight', dpi=300)
            plt.close()
            
            # --- PLOT: MEAN ABSOLUTE JERK ---
            plt.figure(figsize=(8, 4))
            plt.plot(range(MAX_STEPS), avg_jerk_over_time, label='Traffic Shepherd', color='#2ca02c') 
            plt.xlabel('Simulation Step', fontsize=12)
            plt.ylabel('Mean Abs Jerk (m/s³)', fontsize=12)
            plt.xlim([0, MAX_STEPS])
            plt.xticks(ticks, [f"{int(t/1000)}k" if t > 0 else "0k" for t in ticks], fontsize=11)
            plt.yticks(fontsize=11)
            plt.minorticks_on()
            plt.tick_params(direction='in', which='both', right=True, top=True)
            plt.legend(frameon=False, loc='upper center', ncol=4, fontsize=11)
            plt.savefig(os.path.join(save_dir, f"jerk_plot_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.png"), bbox_inches='tight', dpi=300)
            plt.close()

            # --- CDF PLOTTING LOGIC: DECELERATION MAGNITUDE ---
            if all_raw_decels:
                sorted_decels = np.sort(all_raw_decels)
                y_vals = np.arange(1, len(sorted_decels) + 1) / len(sorted_decels)
                
                plt.figure(figsize=(6, 4))
                plt.plot(sorted_decels, y_vals, color='#d62728', linewidth=2)
                plt.axvline(np.mean(all_raw_decels), color='black', linestyle='--', label=f'Mean ({np.mean(all_raw_decels):.2f})')
                
                plt.xlabel('Deceleration Magnitude ($m/s^2$)', fontsize=12)
                plt.ylabel('Cumulative Probability (CDF)', fontsize=12)
                plt.title(f'CDF of CAV Decelerations ({lane_demand} vph | PR {int(pr*100)}%)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                cdf_filename = os.path.join(save_dir, f"cdf_decel_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.png")
                plt.savefig(cdf_filename, bbox_inches='tight', dpi=300)
                plt.close()
                
            # --- CDF PLOTTING LOGIC: INFERENCE TIMES ---
            if all_raw_inference_times:
                sorted_inf = np.sort(all_raw_inference_times)
                y_vals_inf = np.arange(1, len(sorted_inf) + 1) / len(sorted_inf)
                
                p90_inf = np.percentile(all_raw_inference_times, 90)
                mean_inf = np.mean(all_raw_inference_times)
                
                plt.figure(figsize=(6, 4))
                plt.plot(sorted_inf, y_vals_inf, color='#1f77b4', linewidth=2)
                plt.axvline(p90_inf, color='orange', linestyle='--', label=f'90th % ({p90_inf:.2f} ms)')
                plt.axvline(mean_inf, color='black', linestyle='--', label=f'Mean ({mean_inf:.2f} ms)')
                
                plt.xlabel('Inference Time (ms)', fontsize=12)
                plt.ylabel('Cumulative Probability (CDF)', fontsize=12)
                plt.title(f'CDF of Per-Step Inference Time ({lane_demand} vph | PR {int(pr*100)}%)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                inf_cdf_filename = os.path.join(save_dir, f"cdf_inference_SHEPHERD_{lane_demand}vph_{int(pr*100)}PR.png")
                plt.savefig(inf_cdf_filename, bbox_inches='tight', dpi=300)
                plt.close()
            else:
                p90_inf = 0.0
                mean_inf = 0.0

            # --- REPORT ---
            tp_hr_m, tp_hr_c = calc_mean_ci(agg_results['hourly_throughput'])
            tp_raw_m, tp_raw_c = calc_mean_ci(agg_results['raw_arrivals'])
            spd_m, spd_c = calc_mean_ci(agg_results['avg_speed'])
            tt_m, tt_c = calc_mean_ci(agg_results['avg_tt'])
            col_m, col_c = calc_mean_ci(agg_results['collision_rate'])
            near_col_m, near_col_c = calc_mean_ci(agg_results['near_collisions'])
            ttc_m, ttc_c = calc_mean_ci(agg_results['ttc_violations'])
            hb_m, hb_c = calc_mean_ci(agg_results['hard_braking'])
            acc_var_m, acc_var_c = calc_mean_ci(agg_results['accel_var'])
            jerk_m, jerk_c = calc_mean_ci(agg_results['jerk'])
            wave_m, wave_c = calc_mean_ci(agg_results['wave_intensity'])
            succ_m, succ_c = calc_mean_ci(agg_results['success_rate'])
            avg_q_m, avg_q_c = calc_mean_ci(agg_results['avg_max_queue'])
            abs_q_m, abs_q_c = calc_mean_ci(agg_results['abs_max_queue'])
            dly_m, dly_c = calc_mean_ci(agg_results['hdv_delay_mean'])
            dly90_m, dly90_c = calc_mean_ci(agg_results['hdv_delay_90th'])
            int_tot_m, int_tot_c = calc_mean_ci(agg_results['interventions_total'])
            int_cav_m, int_cav_c = calc_mean_ci(agg_results['interventions_per_cav'])
            duty_m, duty_c = calc_mean_ci(agg_results['intervention_duty_cycle'])
            md_m, md_c = calc_mean_ci(agg_results['mean_decel'])
            dur_m, dur_c = calc_mean_ci(agg_results['mean_duration'])

            report = f"""
==================================================
TRAFFIC SHEPHERD: {lane_demand} veh/h/lane | {pr*100:.0f}% PR (Over {len(test_seeds)} Seeds)
Values reported as: Mean ± 95% Confidence Interval
==================================================
[Efficiency]
  Equivalent Hourly Throughput: {tp_hr_m:.1f} ± {tp_hr_c:.1f} veh/h
  Raw Arrivals (400s Episode):  {tp_raw_m:.1f} ± {tp_raw_c:.1f} vehicles
  Mean Speed:                   {spd_m:.2f} ± {spd_c:.2f} m/s
  Mean Travel Time:             {tt_m:.2f} ± {tt_c:.2f} s

[Safety]
  Collision Rate:               {col_m:.2f} ± {col_c:.2f} %
  Avg Near-Collisions (TTC<1.5s): {near_col_m:.1f} ± {near_col_c:.1f} per episode
  Avg TTC Violations (TTC<2.5s): {ttc_m:.1f} ± {ttc_c:.1f} per episode
  Avg Hard Braking:             {hb_m:.1f} ± {hb_c:.1f} per episode

[Driving Comfort (Stability)]
  Acceleration Var:             {acc_var_m:.4f} ± {acc_var_c:.4f} m/s^2
  Mean Abs Jerk:                {jerk_m:.4f} ± {jerk_c:.4f} m/s^3
  Wave Intensity (Var):         {wave_m:.2f} ± {wave_c:.2f} (speed variance)
  
[Control Cost (CAV Only)]
  Raw Interventions (Sum):      {int_tot_m:.0f} ± {int_tot_c:.0f} steps
  Interventions Per CAV:        {int_cav_m:.1f} ± {int_cav_c:.1f} steps/CAV
  Duty Cycle (% Active Control): {duty_m:.2f} ± {duty_c:.2f} %
  Mean Decel Magnitude:         {md_m:.2f} ± {md_c:.2f} m/s^2
  Mean Duration:                {dur_m:.2f} ± {dur_c:.2f} s

[Merge Success & Queue]
  Success Rate:                 {succ_m:.2f} ± {succ_c:.2f} %
  Avg Max Queue:                {avg_q_m:.2f} ± {avg_q_c:.2f} vehicles
  Absolute Max Queue:           {abs_q_m:.1f} ± {abs_q_c:.1f} vehicles

[Fairness (HDV Only)]
  Mean Delay:                   {dly_m:.2f} ± {dly_c:.2f} s
  90th-Percentile Delay:        {dly90_m:.2f} ± {dly90_c:.2f} s
  
[Computational Latency]
The cumulative distribution function (CDF) of the per-step inference time indicates that in 90% of cases, the inference time remains below {p90_inf:.2f} ms, with an average delay of {mean_inf:.2f} ms. Overall, the observed computational latency is sufficiently low to support real-time decision-making in high-speed traffic environments.
==================================================
"""
            print(report, flush=True)
            with open(os.path.join(save_dir, f"report_SHEPHERD_ld{lane_demand}vph_{int(pr*100)}PR.txt"), "w") as file:
                file.write(report)

if __name__ == "__main__":
    main()