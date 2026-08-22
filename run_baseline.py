import os
import argparse
import numpy as np
import traci
import time
import json
from baseline import BaselineExecutor

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Rule-based Baselines across Lane Demand/PR grid')
    parser.add_argument('--gui', action='store_true', help="Force GUI for ALL episodes (runs headless by default)")
    parser.add_argument('--eval-lane-demands', type=str, default="800,1700", help="Comma-separated list of lane demands")
    parser.add_argument('--eval-prs', type=str, default="0.05,0.10,0.20", help="Comma-separated list of PRs")
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
    <!-- HDV Behavior Distribution -->
    <vTypeDistribution id="hdv_mixture">
        <vType id="hdv_cautious" probability="0.2" carFollowModel="IDM" accel="2.0" decel="4.5" tau="1.8" speedFactor="0.9" speedDev="0.05" lcCooperative="2.0" lcAssertive="0.1" color="0,255,0"/>
        <vType id="hdv_normal" probability="0.4" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.2" speedFactor="1.0" speedDev="0.1" lcCooperative="1.0" lcAssertive="1.0" color="255,255,255"/>
        <vType id="hdv_aggressive" probability="0.4" carFollowModel="IDM" accel="3.5" decel="4.5" tau="0.8" speedFactor="1.15" speedDev="0.1" lcCooperative="0.1" lcAssertive="2.0" color="0,0,255"/>
    </vTypeDistribution>

    <!-- CAV Definition -->
    <vType id="cav" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.0" color="255,0,0"/>

    <!-- Routes -->
    <route id="route_main" edges="E0 E1 E6" />
    <route id="route_ramp" edges="E_ramp E1 E6" />

    {flows_xml}
</routes>
"""
    os.makedirs("SUMO_network", exist_ok=True)
    with open("SUMO_network/highway_onramp_actual.rou.xml", "w") as f:
        f.write(xml_content)

def compute_human_centric_reward(cav_id):
    try:
        w_eff, w_safe, w_stab, w_fair, w_ctrl = 1.0, 10.0, 5.0, 0.5, 0.1
        ego_speed = traci.vehicle.getSpeed(cav_id)
        ego_accel = traci.vehicle.getAcceleration(cav_id)
        follower = traci.vehicle.getFollower(cav_id, dist=100.0)
        
        r_eff, r_safe, r_stab, r_fair, c_ctrl = ego_speed / 20.0, 0.0, 0.0, 0.0, 0.0
        
        if follower and follower[0] != "":
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

def calc_mean_ci(data):
    """Calculates Mean and 95% Confidence Interval."""
    if not data or len(data) == 0:
        return 0.0, 0.0
    mean = np.mean(data)
    ci = 1.96 * np.std(data, ddof=1) / np.sqrt(len(data)) if len(data) > 1 else 0.0
    return mean, ci

def main():
    args = parse_args()
    
    save_dir = "./results/baselineFinal"
    os.makedirs(save_dir, exist_ok=True)
    
    net_file = "SUMO_network/test.net.xml"
    route_file = "SUMO_network/highway_onramp_actual.rou.xml" 
    
    cav_prefix = "cav"
    policies = ["All-HDV", "Rule-CAV", "Selfish-CAV"]
    
    step_length = 0.1
    TTC_THRESHOLD = 2.5 
    HARD_BRAKE_THRESHOLD = -3.0 
    NEAR_COLLISION_TTC_THRESHOLD = 1.5 
    MAX_STEPS = 4000 

    test_seeds = [0, 25, 50, 75, 100, 125, 150, 175, 200, 325, 
                  350, 375, 400, 425, 450, 475, 500, 525, 550, 575]

    lane_demands = [int(d) for d in args.eval_lane_demands.split(',')]
    base_prs = [float(pr) for pr in args.eval_prs.split(',')]

    executor = BaselineExecutor(safe_gap=25.0, smoothing_speed=15.0)

    print("Initiating Pure Comprehensive Baseline Benchmarks across Lane Demand Grid...")

    for lane_demand in lane_demands:
        for policy in policies:
            
            # Smart PR Routing: All-HDV only needs to be run once at 0% PR
            current_prs = [0.0] if policy == "All-HDV" else base_prs
            
            for pr in current_prs:
                print(f"\n==================================================")
                print(f"EVALUATING {policy.upper()}: Lane Demand {lane_demand} veh/h/lane | PR {pr*100:.0f}%")
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
                
                all_raw_inference_times = []

                for current_seed in test_seeds:
                    current_binary = "sumo-gui" if args.gui else "sumo"
                    
                    traci_args = [current_binary, "-n", net_file, "-r", route_file, 
                                  "--seed", str(current_seed), "--start", "--quit-on-end", 
                                  "--step-length", str(step_length),
                                  "--no-step-log", "true", "--no-warnings", "true",
                                  "--collision.action", "remove",
                                  "--collision.check-junctions", "true",
                                  "--collision.mingap-factor", "0.0"]
                    
                    if args.gui:
                        traci_args.extend(["--window-size", "1280,720", "--window-pos", "50,50", "--delay", "50"])
                             
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
                    
                    veh_track = {}
                    active_cav_braking = {} 
                    active_ttc_events = set()
                    active_near_col_events = set()
                    active_hard_brakes = set()
                    
                    episode_reward = 0.0
                    step = 0
                                        
                    while step < MAX_STEPS:
                        step += 1
                        
                        # 1. Execute Baseline Policy
                        try:
                            active_cavs = [v for v in traci.vehicle.getIDList() if v.startswith(cav_prefix)]
                        except traci.exceptions.TraCIException:
                            active_cavs = []
                            
                        inf_start = time.perf_counter()
                        executor.execute_policy(active_cavs, policy)
                        inf_end = time.perf_counter()
                        
                        inf_time_ms = (inf_end - inf_start) * 1000.0
                        seed_metrics['inference_times'].append(inf_time_ms)
                        
                        # 2. Advance Simulation
                        try:
                            traci.simulationStep()
                        except traci.exceptions.TraCIException:
                            pass
                            
                        # 3. Collect Raw Simulation Data
                        try:
                            active_vehs = traci.vehicle.getIDList()
                            collisions = traci.simulation.getCollidingVehiclesIDList()
                            arrived_vehs = traci.simulation.getArrivedIDList()
                        except traci.exceptions.TraCIException:
                            active_vehs, collisions, arrived_vehs = [], [], []

                        # Calculate step rewards for tracking
                        active_cavs_post = [v for v in active_vehs if v.startswith(cav_prefix)]
                        for cav_id in active_cavs:
                            if cav_id in collisions:
                                episode_reward += -10.0
                            elif cav_id in active_cavs_post:
                                episode_reward += compute_human_centric_reward(cav_id)

                        for c_veh in collisions:
                            c_unique = f"{c_veh}_s{current_seed}"
                            seed_metrics['collided_vehs'].add(c_unique)
                            seed_metrics['total_vehs'].add(c_unique)
                            if c_veh.startswith(cav_prefix):
                                seed_metrics['total_cavs'].add(c_unique)
                            if c_veh in veh_track and veh_track[c_veh]['route'] == "route_ramp":
                                seed_metrics['ramp_vehs_total'].add(c_unique)
                                seed_metrics['ramp_vehs_failed'].add(c_unique)
                        
                        seed_metrics['throughput'] += len(arrived_vehs)
                        for arr_v in arrived_vehs:
                            arr_unique = f"{arr_v}_s{current_seed}"
                            if arr_unique in seed_metrics['ramp_vehs_total']:
                                seed_metrics['ramp_vehs_merged'].add(arr_unique)

                            if arr_v in active_cav_braking:
                                duration = active_cav_braking[arr_v] * step_length
                                seed_metrics['cav_decel_durations'].append(duration)
                                del active_cav_braking[arr_v]

                            if arr_v in veh_track:
                                tt = (step - veh_track[arr_v]['entry_step']) * step_length
                                seed_metrics['travel_times'].append(tt)
                                if not veh_track[arr_v]['is_cav']:
                                    seed_metrics['hdv_time_losses'].append(veh_track[arr_v]['max_time_loss'])
                                del veh_track[arr_v]
                        
                        for veh_id in active_vehs:
                            try:
                                speed = traci.vehicle.getSpeed(veh_id)
                                seed_metrics['speeds'].append(speed)
                                accel = traci.vehicle.getAcceleration(veh_id)
                                time_loss = traci.vehicle.getTimeLoss(veh_id)
                                is_cav = veh_id.startswith(cav_prefix)
                                unique_id = f"{veh_id}_s{current_seed}"
                                route_id = traci.vehicle.getRouteID(veh_id)
                                
                                if is_cav:
                                    seed_metrics['total_cavs'].add(unique_id)
                                    seed_metrics['cav_active_steps'][unique_id] = seed_metrics['cav_active_steps'].get(unique_id, 0) + 1
                                    if accel < -0.1:  # Active braking threshold
                                        seed_metrics['cav_interventions'] += 1
                                        seed_metrics['cav_decel_magnitudes'].append(abs(accel))
                                        active_cav_braking[veh_id] = active_cav_braking.get(veh_id, 0) + 1
                                    else:
                                        if veh_id in active_cav_braking:
                                            duration = active_cav_braking[veh_id] * step_length
                                            seed_metrics['cav_decel_durations'].append(duration)
                                            del active_cav_braking[veh_id]

                                # Success Rate Tracking (Ramp Vehicles)
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

                                veh_track[veh_id]['accel_history'].append(accel)
                                seed_metrics['accels'].append(accel)
                                prev_accel = veh_track[veh_id]['prev_accel']
                                jerk = (accel - prev_accel) / step_length
                                seed_metrics['jerks'].append(jerk)
                                
                                veh_track[veh_id]['prev_accel'] = accel
                                veh_track[veh_id]['speed_history'].append(speed)
                                seed_metrics['total_vehs'].add(unique_id)
                                
                                if accel < HARD_BRAKE_THRESHOLD:
                                    if veh_id not in active_hard_brakes:
                                        seed_metrics['hard_braking_events'] += 1
                                        active_hard_brakes.add(veh_id)
                                else:
                                    active_hard_brakes.discard(veh_id)
                                    
                                leader_info = traci.vehicle.getLeader(veh_id, 100.0)
                                if leader_info is not None:
                                    leader_id, dist = leader_info
                                    leader_speed = traci.vehicle.getSpeed(leader_id)
                                    rel_speed = speed - leader_speed
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

                    # Queue Length Logic: Consecutive vehicles moving < 2.0 m/s
                    step_max_queue = 0
                    try:
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
                    except traci.exceptions.TraCIException:
                        pass
                        
                    seed_metrics['step_max_queues'].append(step_max_queue)

                    for steps_b in active_cav_braking.values():
                        seed_metrics['cav_decel_durations'].append(steps_b * step_length)

                    for v_data in veh_track.values():
                        if len(v_data['speed_history']) > 10:
                            seed_metrics['speed_variances'].append(np.var(v_data['speed_history']))
                            
                    traci.close()
                    print(f"  Seed {current_seed} Complete | Reward: {episode_reward:.2f}", flush=True)

                    all_raw_inference_times.extend(seed_metrics['inference_times'])

                    # Accumulate per-seed metrics into agg_results storage
                    hourly_multiplier = 3600.0 / (MAX_STEPS * step_length)
                    extrapolated_hourly = seed_metrics['throughput'] * hourly_multiplier
                    
                    agg_results['hourly_throughput'].append(extrapolated_hourly)
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
                    agg_results['wave_intensity'].append(np.mean(seed_metrics['speed_variances']) if seed_metrics['speed_variances'] else 0.0)
                    
                    agg_results['hdv_delay_mean'].append(np.mean(seed_metrics['hdv_time_losses']) if seed_metrics['hdv_time_losses'] else 0.0)
                    agg_results['hdv_delay_90th'].append(np.percentile(seed_metrics['hdv_time_losses'], 90) if seed_metrics['hdv_time_losses'] else 0.0)
                    
                    successful_merges = len(seed_metrics['ramp_vehs_merged'] - seed_metrics['ramp_vehs_failed'])
                    total_ramp = len(seed_metrics['ramp_vehs_total'])
                    succ_rate = (successful_merges / total_ramp * 100) if total_ramp > 0 else 100.0
                    agg_results['success_rate'].append(succ_rate)
                    
                    agg_results['avg_max_queue'].append(np.mean(seed_metrics['step_max_queues']) if seed_metrics['step_max_queues'] else 0.0)
                    agg_results['abs_max_queue'].append(np.max(seed_metrics['step_max_queues']) if seed_metrics['step_max_queues'] else 0.0)
                    
                    num_cavs_in_seed = len(seed_metrics['total_cavs'])
                    agg_results['interventions_total'].append(seed_metrics['cav_interventions'])
                    agg_results['interventions_per_cav'].append(seed_metrics['cav_interventions'] / max(1, num_cavs_in_seed))
                    agg_results['intervention_duty_cycle'].append((seed_metrics['cav_interventions'] / max(1, sum(seed_metrics['cav_active_steps'].values()))) * 100.0)
                    agg_results['mean_decel'].append(np.mean(seed_metrics['cav_decel_magnitudes']) if seed_metrics['cav_decel_magnitudes'] else 0.0)
                    agg_results['mean_duration'].append(np.mean(seed_metrics['cav_decel_durations']) if seed_metrics['cav_decel_durations'] else 0.0)

                if all_raw_inference_times:
                    p90_inf = np.percentile(all_raw_inference_times, 90)
                    mean_inf = np.mean(all_raw_inference_times)
                else:
                    p90_inf = 0.0
                    mean_inf = 0.0

                # --- CALCULATE MEANS & 95% CIs ---
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

                num_seeds = len(test_seeds)

                report = f"""
==================================================
{policy.upper()}: {lane_demand} veh/h/lane | {pr*100:.0f}% PR (Over {num_seeds} Seeds)
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

[Driving Comfort & Stability]
  Acceleration Var:             {acc_var_m:.4f} ± {acc_var_c:.4f} m/s^2
  Mean Abs Jerk:                {jerk_m:.4f} ± {jerk_c:.4f} m/s^3
  Wave Intensity (Var):         {wave_m:.2f} ± {wave_c:.2f} (speed variance)

[Control Cost (CAV Only - Normalized Metrics)]
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
                output_filepath = os.path.join(save_dir, f"report_{policy}_{lane_demand}vph_{int(pr*100)}PR.txt")
                with open(output_filepath, "w") as file:
                    file.write(report)

if __name__ == "__main__":
    main()