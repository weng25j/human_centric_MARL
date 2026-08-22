import traci
import numpy as np

class BaselineExecutor:
    """
    Executes the three deterministic heuristic baselines for the Traffic Shepherd evaluation.
    Includes visual color-coding for the SUMO GUI to track active policies.
    """
    def __init__(self, safe_gap=25.0, smoothing_speed=15.0):
        # Mathematical thresholds for Rule-CAV
        self.safe_gap = safe_gap
        self.smoothing_speed = smoothing_speed

    def execute_policy(self, active_cavs, policy_name):
        """
        Routes the active CAVs to the correct deterministic logic.
        """
        if policy_name == "All-HDV":
            self._apply_all_hdv(active_cavs)
        elif policy_name == "Rule-CAV":
            self._apply_rule_cav(active_cavs)
        elif policy_name == "Selfish-CAV":
            self._apply_selfish_cav(active_cavs)

    def _apply_all_hdv(self, active_cavs):
        """
        Policy 1: All-HDV
        No autonomous intervention. CAVs act exactly like background traffic.
        """
        for cav_id in active_cavs:
            try:
                # -1 returns speed control entirely to SUMO's internal car-following model (Krauss/Gipps)
                traci.vehicle.setSpeed(cav_id, -1)
                
                # Color: White (Standard HDV)
                traci.vehicle.setColor(cav_id, (255, 255, 255, 255))
            except traci.exceptions.TraCIException:
                pass

    def _apply_rule_cav(self, active_cavs):
        """
        Policy 2: Rule-CAV
        Deterministic gap creation and speed smoothing to act as a naive shockwave absorber.
        """
        for cav_id in active_cavs:
            try:
                # Color: Blue (Cooperative/Structured)
                traci.vehicle.setColor(cav_id, (0, 100, 255, 255))
                
                leader = traci.vehicle.getLeader(cav_id, dist=100.0)
                
                if leader is not None:
                    dist_to_leader = leader[1]
                    
                    # Logic A: Gap Creation
                    if dist_to_leader < self.safe_gap:
                        current_speed = traci.vehicle.getSpeed(cav_id)
                        # Decelerate smoothly by 2.0 m/s to open the gap
                        target_speed = max(0.0, current_speed - 2.0)
                        traci.vehicle.setSpeed(cav_id, target_speed)
                        continue
                
                # Logic B: Speed Smoothing (Pace-making)
                # If the gap is safe, maintain a constant stabilizing speed
                traci.vehicle.setSpeed(cav_id, self.smoothing_speed)
                
            except traci.exceptions.TraCIException:
                pass

    def _apply_selfish_cav(self, active_cavs):
        """
        Policy 3: Selfish-CAV
        Optimizes strictly for its own travel time, ignoring following vehicles.
        """
        for cav_id in active_cavs:
            try:
                # Color: Red (Aggressive/Selfish)
                traci.vehicle.setColor(cav_id, (255, 0, 0, 255))
                
                # Force the vehicle to always attempt to drive at the maximum allowed lane speed
                max_speed = traci.vehicle.getAllowedSpeed(cav_id)
                
                # Note: SUMO will still apply emergency braking to prevent physical collisions, 
                # but the vehicle will aggressively tailgate to maintain max_speed.
                traci.vehicle.setSpeed(cav_id, max_speed)
                
            except traci.exceptions.TraCIException:
                pass