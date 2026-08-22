import ray
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from marl_on_ramp_env import MARLOnRampEnv

def env_creator(config):
    """Creates and wraps the custom PettingZoo environment for RLlib."""
    env = MARLOnRampEnv(
        net_file='merging_network.net.xml',
        route_file='traffic_routes.rou.xml',
        use_gui=False, # Disable GUI for much faster training speeds
        max_steps=1000
    )
    return ParallelPettingZooEnv(env)

def main():
    # Initialize the Ray cluster locally
    ray.init(ignore_reinit_error=True)

    # Register the custom environment with Ray's registry
    env_name = "marl_on_ramp_v0"
    register_env(env_name, env_creator)

    print("Configuring the PPO Multi-Agent algorithm...")
    config = (
        PPOConfig()
        .environment(env=env_name)
        # Use 1 worker for standard local Windows setups
        .rollouts(num_rollout_workers=1) 
        .multi_agent(
            # Tells RLlib to count steps based on environment steps, not individual agent steps
            count_steps_by="env_steps",
        )
    )

    # Build the algorithm
    algo = config.build()

    print("Starting training loop...")
    # Train the algorithm for 10 iterations as a test
    for i in range(10):
        result = algo.train()
        
        # Safely extract the mean reward for the iteration
        mean_reward = result.get('env_runners', {}).get('episode_reward_mean', "N/A")
        print(f"Training Iteration {i + 1} | Mean Reward: {mean_reward}")

    algo.stop()
    ray.shutdown()
    print("Training finished successfully.")

if __name__ == "__main__":
    main()