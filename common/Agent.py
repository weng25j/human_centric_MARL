import torch as th
from common.Memory import ReplayMemory

class Agent(object):
    """
    A unified agent interface:
    - interact: interact with the environment to collect experience
    - train: train on a sample batch
    - exploration_action: choose an action based on state with random noise
    - action: choose an action based on state for execution
    - value: evaluate value for a state-action pair
    - evaluation: evaluation a learned agent
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=10000,
                 reward_gamma=0.99, reward_scale=1., done_penalty=None,
                 actor_hidden_size=32, critic_hidden_size=32, critic_loss="mse",
                 actor_lr=0.01, critic_lr=0.01,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True):
        
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # REMOVED: self.env_state, self.action_mask = self.env.reset()
        # REMOVED: self.n_agents = len(self.env.controlled_vehicles)
        # Reason: Traffic Shepherd handles a fluctuating number of CAVs. 
        # State tracking is now handled dynamically inside the explore() method.

        self.n_episodes = 1
        self.n_steps = 0
        self.max_steps = max_steps
        self.roll_out_n_steps = 1

        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.done_penalty = done_penalty

        self.memory = ReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.target_tau = 0.01

        self.use_cuda = use_cuda and th.cuda.is_available()