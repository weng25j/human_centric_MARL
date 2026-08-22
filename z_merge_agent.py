import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque

class ZMergeParameterNetwork(nn.Module):
    def __init__(self, state_dim, num_discrete_actions, param_dim_per_action):
        super().__init__()
        # 4 Hidden layers as defined in Table 1
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.param_out = nn.Linear(128, num_discrete_actions * param_dim_per_action)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return torch.tanh(self.param_out(x))

class ZMergeQNetwork(nn.Module):
    def __init__(self, state_dim, total_param_dim, num_discrete_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + total_param_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.q_out = nn.Linear(128, num_discrete_actions)
        
    def forward(self, state, action_parameters):
        x = torch.cat([state, action_parameters], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.q_out(x)

class ReplayBuffer:
    def __init__(self, capacity=100000): # Updated to 100,000 capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, discrete_action, continuous_params, reward, next_state, done):
        self.buffer.append((state, discrete_action, continuous_params, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, d_action, c_params, reward, next_state, done = map(np.stack, zip(*batch))
        return (torch.FloatTensor(state), torch.LongTensor(d_action), 
                torch.FloatTensor(c_params), torch.FloatTensor(reward), 
                torch.FloatTensor(next_state), torch.FloatTensor(done))
        
    def __len__(self):
        return len(self.buffer)

class ZMergeAgent:
    def __init__(self, state_dim):
        self.num_discrete = 5 # Actions a0 through a4
        self.param_per_action = 1 # 1 continuous parameter bounded [-1, 1] per action
        self.total_param_dim = self.num_discrete * self.param_per_action
        
        self.q_net = ZMergeQNetwork(state_dim, self.total_param_dim, self.num_discrete)
        self.param_net = ZMergeParameterNetwork(state_dim, self.num_discrete, self.param_per_action)
        
        self.q_target = ZMergeQNetwork(state_dim, self.total_param_dim, self.num_discrete)
        self.param_target = ZMergeParameterNetwork(state_dim, self.num_discrete, self.param_per_action)
        
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.param_target.load_state_dict(self.param_net.state_dict())
        
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.param_optimizer = optim.Adam(self.param_net.parameters(), lr=1e-4)
        self.memory = ReplayBuffer()
        self.gradient_steps = 0
        
    def act(self, state, epsilon=0.01):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            continuous_params = self.param_net(state_tensor)
            
            if random.random() < epsilon:
                discrete_action = random.randint(0, self.num_discrete - 1)
            else:
                q_values = self.q_net(state_tensor, continuous_params)
                discrete_action = torch.argmax(q_values).item()
                
        return discrete_action, continuous_params.squeeze(0).numpy()

    def train_step(self, batch_size=64, gamma=0.995):
        if len(self.memory) < batch_size:
            return 0.0, 0.0
            
        states, d_actions, c_params, rewards, next_states, dones = self.memory.sample(batch_size)
        
        with torch.no_grad():
            next_params = self.param_target(next_states)
            next_q_values = self.q_target(next_states, next_params)
            best_next_actions = torch.argmax(next_q_values, dim=1, keepdim=True)
            target_q = next_q_values.gather(1, best_next_actions).squeeze(1)
            y_t = rewards + gamma * target_q * (1 - dones)
            
        current_q = self.q_net(states, c_params).gather(1, d_actions.unsqueeze(1)).squeeze(1)
        
        # Hubber Loss for Q-Network
        q_loss = F.huber_loss(current_q, y_t)
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        predicted_params = self.param_net(states)
        q_values_for_params = self.q_net(states, predicted_params)
        param_loss = -q_values_for_params.sum(dim=1).mean() 
        
        self.param_optimizer.zero_grad()
        param_loss.backward()
        self.param_optimizer.step()
        
        self.gradient_steps += 1
        
        # Hard update every 35,000 steps
        if self.gradient_steps % 35000 == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())
            self.param_target.load_state_dict(self.param_net.state_dict())
            
        return q_loss.item(), param_loss.item()

    # ... (rest of your ZMergeAgent code) ...

    def save(self, filepath="zmerge_model"):
        """Saves the primary networks to your hard drive."""
        torch.save(self.q_net.state_dict(), f"{filepath}_qnet.pth")
        torch.save(self.param_net.state_dict(), f"{filepath}_paramnet.pth")
        print(f"Models saved successfully to {filepath}_qnet.pth and {filepath}_paramnet.pth")

    def load(self, filepath="zmerge_model"):
        """Loads previously trained networks from your hard drive."""
        self.q_net.load_state_dict(torch.load(f"results_zmerge/{filepath}_qnet.pth"))
        self.param_net.load_state_dict(torch.load(f"results_zmerge/{filepath}_paramnet.pth"))
        
        # Copy the loaded weights to the target networks too!
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.param_target.load_state_dict(self.param_net.state_dict())
        print(f"Models loaded successfully from {filepath}!")