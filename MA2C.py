import torch as th
import os, logging
import numpy as np
from torch import nn
from torch.optim import Adam, RMSprop

# Assuming these are imported from your common library:
from common.Agent import Agent
from common.Model import ActorNetwork, CriticNetwork, ActorCriticNetwork
from common.utils import entropy, to_tensor_var

class TrafficShepherdMA2C(Agent):
    """
    Adapted Multi-Agent Advantage Actor-Critic for Dynamic PettingZoo Environments.
    Modified to support the sparse, fluctuating CAV populations required by the 
    Traffic Shepherd mixed-autonomy plan.
    """

    def __init__(self, env, state_dim, action_dim,
                 memory_capacity=10000, max_steps=1000,
                 reward_gamma=0.99, reward_scale=1.,
                 actor_hidden_size=64, critic_hidden_size=64,
                 actor_lr=0.001, critic_lr=0.001,
                 optimizer_type="adam", entropy_reg=0.01,
                 max_grad_norm=0.5, batch_size=100,
                 use_cuda=False):
        
        # Initialize base agent parameters using explicit keyword arguments
        # to prevent variables from shifting into the wrong slots.
        super().__init__(
            env=env, 
            state_dim=state_dim, 
            action_dim=action_dim, 
            memory_capacity=memory_capacity, 
            max_steps=max_steps,
            reward_gamma=reward_gamma, 
            reward_scale=reward_scale, 
            actor_hidden_size=actor_hidden_size, 
            critic_hidden_size=critic_hidden_size,
            actor_lr=actor_lr, 
            critic_lr=critic_lr, 
            optimizer_type=optimizer_type, 
            entropy_reg=entropy_reg,
            max_grad_norm=max_grad_norm, 
            batch_size=batch_size, 
            use_cuda=use_cuda
        )

        self.action_dim = action_dim # 5 high-level interaction roles
        self.state_dim = state_dim   # 4 local sensor inputs
        
        # Initialize the neural networks
        self.actors = ActorNetwork(self.state_dim, self.actor_hidden_size, self.action_dim)
        self.critics = CriticNetwork(self.state_dim, self.critic_hidden_size, 1)

        if optimizer_type == "adam":
            self.actor_optimizers = Adam(self.actors.parameters(), lr=self.actor_lr)
            self.critic_optimizers = Adam(self.critics.parameters(), lr=self.critic_lr)
        
        if self.use_cuda:
            self.actors.cuda()
            self.critics.cuda()

        self.episode_rewards = []
        self.n_steps = 0

    def explore(self):
        """
        Interacts with the PettingZoo environment to collect experience.
        Handles dynamic dictionaries where CAVs enter and leave[cite: 5].
        """
        obs_dict, infos = self.env.reset()
        
        states, actions, rewards, next_states, dones = [], [], [], [], []
        episode_reward = 0
        
        # Step through the environment until the max_steps truncation is reached
        for _ in range(self.max_steps):
            active_agents = list(obs_dict.keys())
            
            # If no CAVs are currently on the road, step the environment forward with empty actions
            if not active_agents:
                next_obs_dict, rewards_dict, term_dict, trunc_dict, infos = self.env.step({})
                obs_dict = next_obs_dict
                if all(trunc_dict.values()) if trunc_dict else False:
                    break
                continue

            # 1. Convert dictionary observations to a batch tensor for active agents
            state_batch = np.array([obs_dict[agent] for agent in active_agents])
            
            # 2. Predict actions using the Actor network
            action_batch = self._predict_actions(state_batch)
            
            # 3. Map the array of actions back to the PettingZoo dictionary format
            action_dict = {agent: act for agent, act in zip(active_agents, action_batch)}
            
            # 4. Step the environment forward
            # This evolves the state based on CAV commands and HDV human behavior[cite: 5].
            next_obs_dict, rewards_dict, term_dict, trunc_dict, infos = self.env.step(action_dict)
            
            # 5. Store transitions for agents that existed in this step
            for agent in active_agents:
                states.append(obs_dict[agent])
                actions.append(action_dict[agent])
                rewards.append(rewards_dict[agent])
                next_states.append(next_obs_dict[agent])
                
                # Check if the agent exited the network or if the episode hit max_steps
                is_done = term_dict[agent] or trunc_dict[agent]
                dones.append(is_done)
                episode_reward += rewards_dict[agent]

            obs_dict = next_obs_dict
            
            # End exploration if the environment sends a global truncation signal
            if all(trunc_dict.values()):
                break

        self.episode_rewards.append(episode_reward)
        
        # Push all collected transitions to the replay buffer
        self.memory.push(states, actions, rewards, next_states, dones)

    def train(self):
        """
        Updates the Actor and Critic networks using the collected experience.
        """
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a flat batch of transitions (agnostic to dynamic agent counts)
        batch = self.memory.sample(self.batch_size)
        
        states_var = to_tensor_var(batch.states, self.use_cuda)
        actions_var = to_tensor_var(batch.actions, self.use_cuda).long()
        rewards_var = to_tensor_var(batch.rewards, self.use_cuda).unsqueeze(1)
        next_states_var = to_tensor_var(batch.next_states, self.use_cuda)
        dones_var = to_tensor_var(batch.dones, self.use_cuda).unsqueeze(1)

        # ---------------------------
        # 1. Update Critic Network
        # ---------------------------
        self.critic_optimizers.zero_grad()
        
        # Calculate TD Target: r + gamma * V(s') * (1 - done)
        values = self.critics(states_var)
        next_values = self.critics(next_states_var).detach()
        target_values = rewards_var + self.reward_gamma * next_values * (1.0 - dones_var)
        
        critic_loss = nn.MSELoss()(values, target_values)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critics.parameters(), self.max_grad_norm)
        self.critic_optimizers.step()

        # ---------------------------
        # 2. Update Actor Network
        # ---------------------------
        self.actor_optimizers.zero_grad()
        
        action_logits = self.actors(states_var) # Shape: [Batch, Action_Dim]
        action_log_probs = th.log_softmax(action_logits, dim=-1)
        
        # Gather the log probabilities of the specific actions taken
        chosen_log_probs = action_log_probs.gather(1, actions_var.unsqueeze(1))
        
        # Calculate Advantage: A(s,a) = Q_target - V(s)
        advantages = (target_values - values.detach())
        
        # Entropy regularization to encourage exploration
        entropy_loss = th.mean(entropy(th.exp(action_log_probs)))
        
        # Policy Gradient Loss
        pg_loss = -th.mean(chosen_log_probs * advantages)
        actor_loss = pg_loss - entropy_loss * self.entropy_reg
        
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actors.parameters(), self.max_grad_norm)
        self.actor_optimizers.step()

    def _predict_actions(self, state_batch):
        """
        Passes a batch of local states through the Actor network to sample discrete actions.
        """
        state_var = to_tensor_var(state_batch, self.use_cuda)
        action_logits = self.actors(state_var)
        action_probs = th.softmax(action_logits, dim=-1)
        
        if self.use_cuda:
            action_probs = action_probs.data.cpu().numpy()
        else:
            action_probs = action_probs.data.numpy()
            
        # Sample an action from the probability distribution for each agent
        actions = [np.random.choice(self.action_dim, p=probs) for probs in action_probs]
        return actions