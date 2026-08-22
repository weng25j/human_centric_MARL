import traci

def control_hdvs_in_merge(traci_conn, hdv_type="hdv_type", ramp_edge="ramp_0", mainline_edge="main_0"):
    """
    Applies custom control logic to HDVs in the merging zone.
    """
    # Retrieve all vehicles currently active in the simulation
    active_vehicles = traci_conn.vehicle.getIDList()
    
    for veh_id in active_vehicles:
        if traci_conn.vehicle.getTypeID(veh_id) == hdv_type:
            current_edge = traci_conn.vehicle.getRoadID(veh_id)
            
            # --- 1. Speed Adjustment (e.g., Gap Creation on Mainline) ---
            if current_edge == mainline_edge:
                # Speed Mode 32: TraCI dictates speed, but SUMO's car-following 
                # model still overrides to prevent rear-end collisions.
                traci_conn.vehicle.setSpeedMode(veh_id, 32) 
                
                current_speed = traci_conn.vehicle.getSpeed(veh_id)
                
                # Example: Command HDV to decelerate slightly to yield
                target_speed = max(0.0, current_speed - 1.5)
                traci_conn.vehicle.setSpeed(veh_id, target_speed)
            
            # --- 2. Forced Lane Changing (e.g., Merging from Ramp) ---
            elif current_edge == ramp_edge:
                # Lane Change Mode 0: Disables SUMO's autonomous strategic and 
                # cooperative lane changes. TraCI gains absolute control.
                traci_conn.vehicle.setLaneChangeMode(veh_id, 0b0000000000)
                
                # Force change to lane 0 (rightmost mainline lane) over 2.0 seconds
                traci_conn.vehicle.changeLane(veh_id, 0, duration=2.0)

import gymnasium as gym
# Assuming your custom on-ramp environment inherits from sumo_rl.SumoEnvironment
from my_custom_env import OnRampEnv 

# Initialize your pre-coded environment
env = OnRampEnv(
    net_file="C:/justin/hk/Traffic_Shepherd/SUMO_network/test.net.xml",
    route_file="C:/justin/hk/Traffic_Shepherd/SUMO_network/merge_test.rou.xml",
    use_gui=True,
    num_seconds=10000
)

obs, info = env.reset()
done = False

while not done:
    # 1. Your RL agent determines the action for CAVs
    action = agent.predict(obs) 
    
    # 2. Inject TraCI commands for HDVs using the active connection
    control_hdvs_in_merge(
        traci_conn=env.sumo, 
        hdv_type="human_driver", 
        ramp_edge="E_ramp", 
        mainline_edge="E_main"
    )
    
    # 3. Step the environment (applies CAV action and advances the simulation)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    obs = next_obs
    done = terminated or truncated

env.close()