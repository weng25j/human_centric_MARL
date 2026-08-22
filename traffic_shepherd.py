import traci
import numpy as np
import collections

class TrafficShepherdEnv:
    """
    The complete Traffic Shepherd Environment integrating Z-Merge (V2X), 
    Trust-MARL (Human Predictability), and MA2C (Spatial Stabilization).
    """
    def __init__(self, net_file, route_file, num_cavs, max_neighbors=3, alpha=0.75):
        self.net_file = net_file
        self.route_file = route_file
        self.num_cavs = num_cavs
        self.cav_prefix = "cav"
        
        # Pillar 3: Chu MA2C Stabilization Parameters
        self.max_neighbors = max_neighbors
        self.alpha = alpha 
        
        # Pillar 2: Trust-MARL Predictability Tracking
        self.history_window = 5
        self.vehicle_speed_history = collections.defaultdict(lambda: collections.deque(maxlen=self.history_window))

    def _get_zmerge_v2x_data(self, cav_id):
        try:
            ego_speed = traci.vehicle.getSpeed(cav_id)
            edge_id = traci.vehicle.getRoadID(cav_id)
            
            try:
                zone_vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
                zone_density = zone_vehicle_count / 10.0 
                zone_speed = traci.edge.getLastStepMeanSpeed(edge_id)
            except traci.exceptions.TraCIException:
                zone_density = 0.0
                zone_speed = ego_speed
                
            return [ego_speed, zone_density, zone_speed]
        except traci.exceptions.TraCIException:
            return [0.0, 0.0, 0.0]

    def _get_trust_marl_variance(self, cav_id):
        try:
            leader = traci.vehicle.getLeader(cav_id, dist=100.0)
            if leader is not None:
                leader_id = leader[0]
                dist_to_leader = leader[1]
                
                current_leader_speed = traci.vehicle.getSpeed(leader_id)
                self.vehicle_speed_history[leader_id].append(current_leader_speed)
                
                if len(self.vehicle_speed_history[leader_id]) == self.history_window:
                    speed_variance = np.var(self.vehicle_speed_history[leader_id])
                else:
                    speed_variance = 0.0
                    
                return [dist_to_leader, speed_variance]
        except traci.exceptions.TraCIException:
            pass
            
        return [100.0, 0.0] 

    def _get_ma2c_fingerprints(self, cav_id, active_cavs, previous_policies):
        try:
            pos_i = np.array(traci.vehicle.getPosition(cav_id))
        except traci.exceptions.TraCIException:
            pos_i = np.array([0.0, 0.0])
            
        others = [c for c in active_cavs if c != cav_id]
        
        def sort_key(c):
            try:
                return np.linalg.norm(pos_i - np.array(traci.vehicle.getPosition(c)))
            except traci.exceptions.TraCIException:
                return float('inf')
                
        others.sort(key=sort_key)
        nearest = others[:self.max_neighbors]
        
        fp_vec = []
        for n in nearest:
            # 3 actions = 3 probabilities
            fp_vec.extend(previous_policies.get(n, [0.33, 0.33, 0.34]))
            
        while len(fp_vec) < (self.max_neighbors * 3):
            fp_vec.extend([0.0] * 3)
            
        return fp_vec

    def get_observations(self, previous_policies):
        try:
            active_cavs = [v for v in traci.vehicle.getIDList() if v.startswith(self.cav_prefix)]
        except traci.exceptions.TraCIException:
            return {}, []
            
        obs_dict = {}
        
        for cav_id in active_cavs:
            zmerge_obs = self._get_zmerge_v2x_data(cav_id)
            trust_obs = self._get_trust_marl_variance(cav_id)
            ma2c_fp = self._get_ma2c_fingerprints(cav_id, active_cavs, previous_policies)
            
            # State dim = 3 (V2X) + 2 (Trust) + 9 (MA2C) = 14
            full_obs = np.array(zmerge_obs + trust_obs + ma2c_fp, dtype=np.float32)
            obs_dict[cav_id] = full_obs
            
        return obs_dict, active_cavs

    def _compute_human_centric_reward(self, cav_id):
        try:
            w_eff = 1.0   
            w_safe = 10.0 
            w_stab = 5.0  
            w_fair = 0.5  
            w_ctrl = 0.1  
            
            ego_speed = traci.vehicle.getSpeed(cav_id)
            ego_accel = traci.vehicle.getAcceleration(cav_id)
            follower = traci.vehicle.getFollower(cav_id, dist=100.0)
            
            r_eff = ego_speed / 20.0 
            r_safe = 0.0
            r_stab = 0.0
            r_fair = 0.0
            c_ctrl = 0.0
            
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
                
            final_reward = (w_eff * r_eff) + \
                           (w_safe * r_safe) + \
                           (w_stab * r_stab) + \
                           (w_fair * r_fair) - \
                           (w_ctrl * c_ctrl)
                           
            return final_reward
            
        except traci.exceptions.TraCIException:
            return 0.0

    def step(self, action_dict):
        """
        Executes 3 discrete semantic roles with dynamic physics-based acceleration
        and explicit safety overrides.
        """
        for cav_id, action in action_dict.items():
            try:
                if action == 0:
                    # 0: Non-Intervention (Return control to native IDM)
                    traci.vehicle.setSpeedMode(cav_id, 31)
                    traci.vehicle.setSpeed(cav_id, -1)
                else:
                    # Disable safety checks so the CAV fully commits to the intervention
                    traci.vehicle.setSpeedMode(cav_id, 0)
                    current_speed = traci.vehicle.getSpeed(cav_id)
                    
                    if action == 1:
                        # 1: Gap Creation (Dynamic Deceleration)
                        leader = traci.vehicle.getLeader(cav_id, 100.0)
                        if leader is not None:
                            leader_speed = traci.vehicle.getSpeed(leader[0])
                            target_speed = max(0.0, leader_speed - 2.0)
                        else:
                            target_speed = max(0.0, current_speed - 2.0)
                            
                        # Calculate required acceleration (dv) and clip to realistic limits
                        dynamic_accel = np.clip(target_speed - current_speed, -3.0, -0.1)
                        traci.vehicle.setAcceleration(cav_id, dynamic_accel, duration=0.1)
                        
                    elif action == 2:
                        # 2: Pace-make / Release (Dynamic Acceleration)
                        edge_id = traci.vehicle.getRoadID(cav_id)
                        try:
                            mean_edge_speed = traci.edge.getLastStepMeanSpeed(edge_id)
                            target_speed = min(25.0, mean_edge_speed + 3.0) 
                        except traci.exceptions.TraCIException:
                            target_speed = min(25.0, current_speed + 2.0)
                            
                        # Calculate required acceleration and clip to comfortable limits
                        dynamic_accel = np.clip(target_speed - current_speed, 0.1, 1.5)
                        traci.vehicle.setAcceleration(cav_id, dynamic_accel, duration=0.1)
                        
            except traci.exceptions.TraCIException:
                pass
                
        traci.simulationStep()
        
        # 1. Grab collision data IMMEDIATELY after the step
        try:
            collided_vehs = set(traci.simulation.getCollidingVehiclesIDList())
        except traci.exceptions.TraCIException:
            collided_vehs = set()
            
        # 2. Get active CAVs (This will NOT include vehicles SUMO just removed for crashing)
        try:
            active_cavs = set([v for v in traci.vehicle.getIDList() if v.startswith(self.cav_prefix)])
        except traci.exceptions.TraCIException:
            active_cavs = set()
            
        raw_rewards = {}
        
        # 3. Calculate rewards based on the actions taken THIS step, not just who survived
        for cav_id in action_dict.keys():
            if cav_id in collided_vehs:
                # HUGE TERMINAL PENALTY for crashing
                raw_rewards[cav_id] = -10.0 
            elif cav_id in active_cavs:
                # Survived normally, calculate human-centric reward
                raw_rewards[cav_id] = self._compute_human_centric_reward(cav_id)
            else:
                # Reached the destination edge naturally and exited safely
                raw_rewards[cav_id] = 0.0
        
        # 4. Compute Spatial Sharing (MA2C)
        spatial_rewards = {}
        for cav_i in action_dict.keys():
            # If the vehicle crashed or exited, it no longer has a position to share.
            # Just give it its raw reward (e.g. the -10.0 penalty) without spatial smoothing.
            if cav_i not in active_cavs:
                spatial_rewards[cav_i] = raw_rewards[cav_i]
                continue
                
            r_tilde = 0.0
            try:
                pos_i = np.array(traci.vehicle.getPosition(cav_i))
                
                for cav_j in active_cavs:
                    pos_j = np.array(traci.vehicle.getPosition(cav_j))
                    dist_m = np.linalg.norm(pos_i - pos_j)
                    
                    d_edges = int(dist_m / 50.0)
                    r_tilde += (self.alpha ** d_edges) * raw_rewards.get(cav_j, 0.0)
            except traci.exceptions.TraCIException:
                r_tilde = raw_rewards[cav_i]
                
            spatial_rewards[cav_i] = r_tilde
            
        return spatial_rewards