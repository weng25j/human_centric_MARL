import traci
import numpy as np
import math

class ZMergeEnv:
    def __init__(self, net_file, route_file, use_gui=False):
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.cav_prefix = "cav"
        self.run_id = 0
        
        # Empirically chosen reward weights
        self.weights = {'e': 2.0, 's': 2.5, 'c': 0.5, 'q': 1.0, 'd': 1.5, 'l': 0.2}

    def reset(self):
        self.prev_accel = {}
        if self.run_id > 0:
            traci.close()
            
        sumo_cmd = [
            "sumo-gui" if self.use_gui else "sumo",
            "-n", self.net_file, "-r", self.route_file, "--random", 
            "--start", "--quit-on-end", "--step-length", "0.1", 
            "--no-step-log", "true", "--no-warnings", "false"
        ]
        traci.start(sumo_cmd)
        self.run_id += 1
        traci.simulationStep()
        
        return self._get_observations()

    def _get_observations(self):
        obs = {}
        vehicles = traci.vehicle.getIDList()
        cavs = [v for v in vehicles if v.startswith(self.cav_prefix)]
        
        # Global RSU Stats (Placeholder values to replace with lane area detectors)
        pre_merge_density = 0.5 
        merge_density = 0.6
        ramp_density = 0.2
        avg_speed_pre = 25.0
        avg_speed_merge = 22.0
        avg_speed_ramp = 15.0
        queue_accel = 2
        queue_main = 5
        
        for cav in cavs:
            try:
                # Local Ego Info
                speed = traci.vehicle.getSpeed(cav)
                accel = traci.vehicle.getAcceleration(cav)
                lane = traci.vehicle.getLaneIndex(cav)
                dist_to_merge = 100.0 # Placeholder for remaining route to merge point
                time_to_merge = dist_to_merge / (speed + 0.001)
                
                # Construct state array matching the paper's formulation
                state_list = [
                    avg_speed_pre, avg_speed_merge, avg_speed_ramp,
                    pre_merge_density, merge_density, ramp_density,
                    queue_accel, queue_main,
                    speed, accel, lane, time_to_merge
                ]
                
                # Neighbor vehicle stats padding for 2 neighbors x 5 features:
                state_list.extend([0.0] * 10) 
                
                obs[cav] = np.array(state_list, dtype=np.float32)
            except traci.exceptions.TraCIException:
                pass
        return obs

    def step(self, actions_dict):
        for agent_id, action_tuple in actions_dict.items():
            discrete_action, continuous_array = action_tuple
            
            # --- We can safely put the try/except back now! ---
            try:
                target_param = continuous_array[discrete_action] 
                
                if discrete_action == 0 or discrete_action == 1:
                    # 1. Safely calculate the absolute target lane
                    current_lane = traci.vehicle.getLaneIndex(agent_id)
                    edge_id = traci.vehicle.getRoadID(agent_id)
                    max_lanes = traci.edge.getLaneNumber(edge_id)
                    
                    if discrete_action == 0:
                        # Change Left (+1), bounded by the max lanes
                        target_lane = min(current_lane + 1, max_lanes - 1)
                    else:
                        # Change Right (-1), bounded by the rightmost lane (0)
                        target_lane = max(current_lane - 1, 0)
                        
                    traci.vehicle.changeLane(agent_id, target_lane, duration=0.1)

                elif discrete_action == 2:
                    # Accelerate/Decelerate mapped to [-4.5, 2.6] m/s^2
                    accel = np.interp(target_param, [-1, 1], [-4.5, 2.6]) 
                    traci.vehicle.setAcceleration(agent_id, accel, duration=0.1)
                    
                elif discrete_action == 3:
                    # Gap adjustment mapped to [5, 20] meters
                    desired_gap = np.interp(target_param, [-1, 1], [5.0, 20.0])
                    traci.vehicle.setMinGap(agent_id, desired_gap)
                    
                elif discrete_action == 4:
                    pass # Maintain state
                    
            except traci.exceptions.TraCIException:
                pass
                
        traci.simulationStep()
        
        next_obs = self._get_observations()
        rewards, dones = {}, {}
        for agent_id in next_obs.keys():
            rewards[agent_id] = self._compute_reward(agent_id)
            dones[agent_id] = False
            
        return next_obs, rewards, dones

    def _compute_reward(self, agent_id):
        """
        Human-Centric Reward Function (Unified for Fair Baseline Comparison).
        Evaluates efficiency, safety, stability, fairness, and control cost.
        """
        try:
            # 1. EXPLICIT WEIGHTS
            w_eff = 1.0   # Efficiency
            w_safe = 10.0 # Safety
            w_stab = 5.0  # Stability
            w_fair = 0.5  # Fairness
            w_ctrl = 0.1  # Control Cost
            
            # 2. GATHER TRACI METRICS
            # FIXED: Changed 'cav_id' to 'agent_id' to match the function parameter
            ego_speed = traci.vehicle.getSpeed(agent_id)
            ego_accel = traci.vehicle.getAcceleration(agent_id)
            follower = traci.vehicle.getFollower(agent_id, dist=100.0)
            
            r_eff = 0.0
            r_safe = 0.0
            r_stab = 0.0
            r_fair = 0.0
            c_ctrl = 0.0
            
            # --- EFFICIENCY ---
            # Measures throughput, mean speed, and travel time
            r_eff = ego_speed / 20.0 
            
            if follower[0] != "":
                follower_id = follower[0]
                follower_accel = traci.vehicle.getAcceleration(follower_id)
                follower_wait = traci.vehicle.getWaitingTime(follower_id)
                
                # --- SAFETY ---
                # Penalizes collisions, low time-to-collision events, and hard braking
                if follower_accel < -4.5:  # Standard TraCI hard-braking threshold
                    r_safe -= 1.0
                    
                # --- STABILITY ---
                # Penalizes acceleration variance, jerk, and oscillation intensity
                if follower_accel < 0.0:
                    # Scales the penalty based on how hard the wave causes the human to brake
                    r_stab -= abs(follower_accel) / 5.0 
                    
                # --- FAIRNESS ---
                # Measures the distribution of HDV delays
                if follower_wait > 0:
                    r_fair -= follower_wait / 100.0
                    
            # --- CONTROL COST ---
            # Discourages unnecessary braking or excessive intervention
            if ego_accel < 0.0:
                c_ctrl += abs(ego_accel) / 5.0
                
            # 3. APPLY FINAL FORMULA
            final_reward = (w_eff * r_eff) + \
                           (w_safe * r_safe) + \
                           (w_stab * r_stab) + \
                           (w_fair * r_fair) - \
                           (w_ctrl * c_ctrl)
                           
            return float(final_reward)
            
        except traci.exceptions.TraCIException:
            return 0.0