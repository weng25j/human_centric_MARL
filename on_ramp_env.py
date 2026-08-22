import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Box
import traci

class MARLOnRampEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "traffic_shepherd_marl_v0"}

    def __init__(self, net_file, route_file, use_gui=False, max_steps=1000):
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.max_steps = max_steps
        self.current_step = 0
        self.run_id = 0
        self.prev_accel = {}
        
        # self.agents tracks vehicles CURRENTLY in the network
        self.agents = []
        
        # Traffic Shepherd assumes a sparse control scenario (e.g., 5-20% penetration)[cite: 5].
        # We use a prefix to identify which vehicles are controlled by the MARL algorithm.
        self.cav_prefix = "cav"

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # [Ego_Speed, Lane_Position, Distance_To_Leader, Leader_Speed]
        # This fulfills the plan's requirement for limited local sensing[cite: 5].
        return Box(low=0.0, high=1000.0, shape=(4,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # Continuous acceleration from -4.5 to 3.0 m/s^2
        # (Note: If you want to use the high-level categorical roles like "Gap Creation", 
        # change this to spaces.Discrete(5) and update the step function logic).
        return Box(low=-4.5, high=3.0, shape=(1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.agents = [] 
        
        if self.run_id > 0:
            traci.close()
            
        sumo_cmd = [
            "sumo-gui" if self.use_gui else "sumo",
            "-n", self.net_file,
            "-r", self.route_file,
            "--start",
            "--quit-on-end",
            "--step-length", "0.1", 
            "--no-step-log", "true",
            "--no-warnings", "false"
        ]
        
        traci.start(sumo_cmd)
        self.run_id += 1
        
        # Step once to initialize the network and spawn the first vehicles
        traci.simulationStep()
        
        # Scan for active agents (CAVs) currently in the map
        all_vehicles = traci.vehicle.getIDList()
        self.agents = [vid for vid in all_vehicles if vid.startswith(self.cav_prefix)]
        
        # Generate initial observations and infos for PettingZoo
        observations = {agent: self._get_obs(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        

        
        return observations, infos

    def step(self, actions):
        self.current_step += 1
        
        # 1. Apply actions ONLY to vehicles currently in the network
        active_vehicles = traci.vehicle.getIDList()
        for agent_id, action in actions.items():
            try:
                # 1. Map the integer action (0-4) to an actual acceleration value (m/s^2)
                # Adjust these numbers based on what your 5 Traffic Shepherd roles actually are!
                if action == 0:
                    accel = -4.5  # Hard brake
                elif action == 1:
                    accel = -1.5  # Gentle brake
                elif action == 2:
                    accel = 0.0   # Coast / Maintain speed
                elif action == 3:
                    accel = 1.5   # Gentle accelerate
                elif action == 4:
                    accel = 3.0   # Hard accelerate
                else:
                    accel = 0.0
                
                # 2. Send the float value to TraCI (No [0] needed anymore!)
                traci.vehicle.setAcceleration(agent_id, accel, duration=0.1)
                
            except traci.exceptions.TraCIException:
                # Ignore if the car already left the network
                pass
                
        # 2. Advance the SUMO physics engine
        # This evolves the state based on CAV commands and HDV human behavior[cite: 5].
        traci.simulationStep()
        
        # 3. DYNAMICALLY SCAN FOR AGENTS
        current_active_vehicles = traci.vehicle.getIDList()
        current_cavs = [vid for vid in current_active_vehicles if vid.startswith(self.cav_prefix)]
        
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}
        
        env_truncated = self.current_step >= self.max_steps
        
        # 4. Process all agents (those active last step + newly spawned agents)
        # We must process agents that just exited so we can return terminations[agent] = True
        all_tracked_agents = set(self.agents) | set(current_cavs)
        
        for agent in all_tracked_agents:
            if agent in current_cavs:
                # Agent is still driving in the network
                observations[agent] = self._get_obs(agent)
                rewards[agent] = self._compute_reward(agent)
                terminations[agent] = False
                truncations[agent] = env_truncated
                infos[agent] = {}
            else:
                # Agent just exited the network (reached its destination)
                observations[agent] = np.zeros(4, dtype=np.float32)
                rewards[agent] = self._compute_reward(agent) # Final exit reward
                terminations[agent] = True
                truncations[agent] = env_truncated
                infos[agent] = {}
                
        # 5. Update self.agents to ONLY those currently active for the next step loop
        self.agents = current_cavs
        
        # Optional: If the environment should end when all agents exit after a certain point
        if not self.agents and self.current_step > 100:
            env_truncated = True
            truncations = {agent: True for agent in truncations.keys()}
        
        return observations, rewards, terminations, truncations, infos

    def _get_obs(self, agent_id):
        """
        Gathers localized sensor data for the RL agent. 
        Traffic Shepherd relies on limited local sensing rather than full global observability[cite: 5].
        """
        try:
            # Ego vehicle data
            speed = traci.vehicle.getSpeed(agent_id)
            lane_pos = traci.vehicle.getLanePosition(agent_id)
            
            # Gather limited local neighborhood data (e.g., the leader vehicle)
            leader_info = traci.vehicle.getLeader(agent_id, dist=100.0)
            
            if leader_info is not None:
                leader_id, distance_to_leader = leader_info
                leader_speed = traci.vehicle.getSpeed(leader_id)
            else:
                # If no car is ahead within 100m, assume free flow
                distance_to_leader = 100.0 
                leader_speed = speed 
                
            return np.array([speed, lane_pos, distance_to_leader, leader_speed], dtype=np.float32)
            
        except traci.exceptions.TraCIException:
            # Fallback if the vehicle just exited the network during this exact step
            return np.zeros(4, dtype=np.float32)

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
        
    def close(self):
        traci.close()