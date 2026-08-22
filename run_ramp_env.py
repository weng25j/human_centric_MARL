from on_ramp_env import MARLOnRampEnv

def main():
    print("Initializing environment...")
    env = MARLOnRampEnv(
        net_file='test.net.xml',
        route_file='highway_onramp_actual.rou.xml',
        use_gui=True, # Set to True to watch the vehicles in SUMO
        max_steps=5000
    )

    observations, infos = env.reset()
    
    step = 0
    # The loop continues as long as there are agents in the network
    while step < env.max_steps:
        step += 1
        
        # Sample a random action for every currently active agent
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Step the simulation forward
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if step % 50 == 0:
            print(f"Step {step}: {len(env.agents)} CAVs currently active.")

    env.close()
    print("Simulation complete! All CAVs have exited the network.")

if __name__ == "__main__":
    main()