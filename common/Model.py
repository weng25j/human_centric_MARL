import torch as th
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """
    A network for the actor (policy)
    """
    def __init__(self, state_dim, hidden_size, output_size):
        super(ActorNetwork, self).__init__()
        
        # Standard fully connected layers adapted to your state_dim (4)
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        
        # Return raw logits. The adapted MA2C agent handles the log_softmax 
        # internally for numerical stability.
        logits = self.fc3(out)
        return logits


class CriticNetwork(nn.Module):
    """
    A network for the critic (value function)
    """
    def __init__(self, state_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        
        # Outputs a single value estimating the expected future reward
        out = self.fc3(out)
        return out


class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shares lower-layer representations but
    has distinct output layers for policy and value.
    (Used if you set shared_network=True in your agent config)
    """
    def __init__(self, state_dim, action_dim, hidden_size, critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Distinct heads for Actor (policy) and Critic (value)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)

    def forward(self, state, out_type='p'):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        
        if out_type == 'p':
            # Return policy logits
            logits = self.actor_linear(out)
            return logits
        else:
            # Return state value
            value = self.critic_linear(out)
            return value