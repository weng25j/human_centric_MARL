import torch as th
from torch import nn
import numpy as np
import os, logging
from copy import deepcopy

# Assuming these are available in your project structure
from common.Memory_common import OnPolicyReplayMemory
from common.Model_common import ActorNetwork, CriticNetwork
from common.utils import index_to_one_hot, to_tensor_var
from torch.optim import Adam, RMSprop

# --- HARDCODED SEED ---
torch_seed = 42
th.manual_seed(torch_seed)
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True

class MAPPO:
    """
    An multi-agent learned with PPO
    Adapted for SUMO/TraCI environments (Dictionary-based states/actions).
    """
    def __init__(self, env, state_dim, action_dim, num_cavs=5,
                 memory_capacity=10000, max_steps=1000,
                 roll_out_n_steps=1, target_tau=1.,
                 target_update_steps=5, clip_param=0.2,
                 reward_gamma=0.99, reward_scale=20,
                 actor_hidden_size=128, critic_hidden_size=128,
                 actor_output_act=nn.functional.log_softmax, critic_loss="mse",
                 actor_lr=0.0001, critic_lr=0.0001, test_seeds=0,
                 optimizer_type="rmsprop", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100, episodes_before_train=100,
                 use_cuda=True, traffic_density=1, reward_type="global_R"):

        assert traffic_density in [1, 2, 3]
        assert reward_type in ["regionalR", "global_R"]
        self.reward_type = reward_type
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # FIXED AGENT COUNT FOR PYTORCH TENSORS
        self.n_agents = num_cavs 
        
        # Safe initialization for TraCI
        self.env_state = np.zeros((self.n_agents, self.state_dim))
        self.active_cavs = []
        self.obs_dict = {}
        
        self.n_episodes = 0
        self.n_steps = 0
        self.max_steps = max_steps
        self.test_seeds = test_seeds
        self.reward_gamma = reward_gamma
        self.reward_scale = reward_scale
        self.traffic_density = traffic_density
        self.memory = OnPolicyReplayMemory(memory_capacity)
        self.actor_hidden_size = actor_hidden_size
        self.critic_hidden_size = critic_hidden_size
        self.actor_output_act = actor_output_act
        self.critic_loss = critic_loss
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.optimizer_type = optimizer_type
        self.entropy_reg = entropy_reg
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.episodes_before_train = episodes_before_train
        self.use_cuda = use_cuda and th.cuda.is_available()
        self.roll_out_n_steps = roll_out_n_steps
        self.target_tau = target_tau
        self.target_update_steps = target_update_steps
        self.clip_param = clip_param

        self.actor = ActorNetwork(self.state_dim, self.actor_hidden_size,
                                  self.action_dim, self.actor_output_act)
        self.critic = CriticNetwork(self.state_dim, self.action_dim, self.critic_hidden_size, 1)
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

        if self.optimizer_type == "adam":
            self.actor_optimizer = Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = Adam(self.critic.parameters(), lr=self.critic_lr)
        elif self.optimizer_type == "rmsprop":
            self.actor_optimizer = RMSprop(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = RMSprop(self.critic.parameters(), lr=self.critic_lr)

        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.episode_rewards = [0]
        self.average_speed = [0]
        self.epoch_steps = [0]

    # agent interact with the environment to collect experience
    def interact(self):
        if (self.max_steps is not None) and (self.n_steps >= self.max_steps):
            # Assuming env has a reset that returns obs_dict
            self.obs_dict, _ = self.env.reset()
            self.n_steps = 0
            
        states = []
        actions = []
        rewards = []
        done = False
        average_speed = 0

        # take n steps
        for i in range(self.roll_out_n_steps):
            # --- PADDING BRIDGE: Dictionary to Array ---
            state_array = np.zeros((self.n_agents, self.state_dim))
            cav_list = list(self.obs_dict.keys())
            
            for idx in range(min(self.n_agents, len(cav_list))):
                state_array[idx] = self.obs_dict[cav_list[idx]]
                
            self.env_state = state_array
            states.append(self.env_state)

            # Get actions array from actor
            action = self.exploration_action(self.env_state, self.n_agents)
            
            # --- PADDING BRIDGE: Array to Dictionary ---
            action_dict = {}
            for idx in range(min(self.n_agents, len(cav_list))):
                # Mapping the discrete action index directly to the CAV ID
                action_dict[cav_list[idx]] = action[idx]

            # Step TraCI Environment
            spatial_rewards_dict = self.env.step(action_dict)
            
            # Get next state and calculate global reward
            next_obs_dict, current_active_cavs = self.env.get_observations({})
            
            global_reward = sum(spatial_rewards_dict.values())
            reward_array = [0.0] * self.n_agents
            for idx in range(min(self.n_agents, len(cav_list))):
                cav_id = cav_list[idx]
                if cav_id in spatial_rewards_dict:
                    reward_array[idx] = spatial_rewards_dict[cav_id]

            actions.append([index_to_one_hot(a, self.action_dim) for a in action])
            self.episode_rewards[-1] += global_reward
            self.epoch_steps[-1] += 1
            
            if self.reward_type == "regionalR":
                reward = reward_array
            elif self.reward_type == "global_R":
                reward = [global_reward] * self.n_agents
                
            rewards.append(reward)
            
            # Update for next loop
            self.obs_dict = next_obs_dict
            self.n_steps += 1
            
            # Check TraCI termination conditions
            if self.n_steps >= self.max_steps:
                done = True
                self.obs_dict, _ = self.env.reset()
                break

        # discount reward
        if done:
            final_value = [0.0] * self.n_agents
            self.n_episodes += 1
            self.episode_done = True
            self.episode_rewards.append(0)
            self.average_speed[-1] = average_speed / max(1, self.epoch_steps[-1])
            self.average_speed.append(0)
            self.epoch_steps.append(0)
        else:
            self.episode_done = False
            # Needs the padded state array for final value
            final_state_array = np.zeros((self.n_agents, self.state_dim))
            next_cav_list = list(self.obs_dict.keys())
            for idx in range(min(self.n_agents, len(next_cav_list))):
                final_state_array[idx] = self.obs_dict[next_cav_list[idx]]
                
            final_action = self.action(final_state_array, self.n_agents)
            final_value = self.value(final_state_array, final_action)

        if self.reward_scale > 0:
            rewards = np.array(rewards) / self.reward_scale

        for agent_id in range(self.n_agents):
            rewards[:, agent_id] = self._discount_reward(rewards[:, agent_id], final_value[agent_id])

        rewards = rewards.tolist()
        self.memory.push(states, actions, rewards)

    # train on a roll out batch
    def train(self):
        if self.n_episodes <= self.episodes_before_train:
            return

        batch = self.memory.sample(self.batch_size)
        states_var = to_tensor_var(batch.states, self.use_cuda).view(-1, self.n_agents, self.state_dim)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).view(-1, self.n_agents, self.action_dim)
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).view(-1, self.n_agents, 1)

        for agent_id in range(self.n_agents):
            # update actor network
            self.actor_optimizer.zero_grad()
            values = self.critic_target(states_var[:, agent_id, :], actions_var[:, agent_id, :]).detach()
            advantages = rewards_var[:, agent_id, :] - values

            action_log_probs = self.actor(states_var[:, agent_id, :])
            action_log_probs = th.sum(action_log_probs * actions_var[:, agent_id, :], 1)
            old_action_log_probs = self.actor_target(states_var[:, agent_id, :]).detach()
            old_action_log_probs = th.sum(old_action_log_probs * actions_var[:, agent_id, :], 1)
            ratio = th.exp(action_log_probs - old_action_log_probs)
            surr1 = ratio * advantages
            surr2 = th.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            
            actor_loss = -th.mean(th.min(surr1, surr2))
            actor_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # update critic network
            self.critic_optimizer.zero_grad()
            target_values = rewards_var[:, agent_id, :]
            values = self.critic(states_var[:, agent_id, :], actions_var[:, agent_id, :])
            if self.critic_loss == "huber":
                critic_loss = nn.functional.smooth_l1_loss(values, target_values)
            else:
                critic_loss = nn.MSELoss()(values, target_values)
            critic_loss.backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()

        if self.n_episodes % self.target_update_steps == 0 and self.n_episodes > 0:
            self._soft_update_target(self.actor_target, self.actor)
            self._soft_update_target(self.critic_target, self.critic)

    # predict softmax action based on state
    def _softmax_action(self, state, n_agents):
        state_var = to_tensor_var([state], self.use_cuda)
        softmax_action = []
        for agent_id in range(n_agents):
            softmax_action_var = th.exp(self.actor(state_var[:, agent_id, :]))
            if self.use_cuda:
                softmax_action.append(softmax_action_var.data.cpu().numpy()[0])
            else:
                softmax_action.append(softmax_action_var.data.numpy()[0])
        return softmax_action

    # choose an action based on state with random noise added for exploration in training
    def exploration_action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # choose an action based on state for execution
    def action(self, state, n_agents):
        softmax_actions = self._softmax_action(state, n_agents)
        actions = []
        for pi in softmax_actions:
            actions.append(np.random.choice(np.arange(len(pi)), p=pi))
        return actions

    # evaluate value for a state-action pair
    def value(self, state, action):
        state_var = to_tensor_var([state], self.use_cuda)
        action = index_to_one_hot(action, self.action_dim)
        action_var = to_tensor_var([action], self.use_cuda)

        values = [0] * self.n_agents
        for agent_id in range(self.n_agents):
            value_var = self.critic(state_var[:, agent_id, :], action_var[:, agent_id, :])
            if self.use_cuda:
                values[agent_id] = value_var.data.cpu().numpy()[0]
            else:
                values[agent_id] = value_var.data.numpy()[0]
        return values

    def _discount_reward(self, rewards, final_value):
        discounted_r = np.zeros_like(rewards)
        running_add = final_value
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.reward_gamma + rewards[t]
            discounted_r[t] = running_add
        return discounted_r

    def _soft_update_target(self, target, source):
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(
                (1. - self.target_tau) * t.data + self.target_tau * s.data)

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = os.path.join(model_dir, save_file)
            checkpoint = th.load(file_path)
            print('Checkpoint loaded: {}'.format(file_path))
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.actor.train()
            else:
                self.actor.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def save(self, model_dir, global_step):
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, 'checkpoint-{:d}.pt'.format(global_step))
        th.save({'global_step': global_step,
                 'model_state_dict': self.actor.state_dict(),
                 'optimizer_state_dict': self.actor_optimizer.state_dict()},
                file_path)