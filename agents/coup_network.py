import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from coup_game import Action


class ObservationProcessor:
    """converts coup environment observations to tensors for neural network"""
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.num_roles = 5  # duke, assassin, captain, ambassador, contessa
        self.num_actions = len(Action)
        
        # calculate flattened observation size
        self.obs_size = self._calculate_obs_size()
    
    def _calculate_obs_size(self) -> int:
        """calculate the total size of flattened observation"""
        size = 0
        
        # public information
        size += self.num_players  # coins
        size += self.num_players * self.num_roles  # lost_influence matrix
        size += self.num_players  # is_eliminated
        
        # private information
        size += self.num_roles  # hand (own cards)
        
        # game state
        size += 1  # current_player (normalized)
        size += 1  # phase (normalized)
        size += 1  # turn_count (normalized)
        
        # action context
        size += 1  # pending_action (normalized)
        size += 1  # pending_target (normalized)
        size += 1  # can_respond
        
        # action history (20 actions with player/target info)
        size += 20 * 3  # action_history, action_players, action_targets
        
        return size
    
    def process_observation(self, obs: Dict) -> torch.Tensor:
        """convert observation dict to flattened tensor"""
        # collect all components
        components = []
        
        # public information - normalize coins to reasonable range
        coins_normalized = obs["coins"] / 20.0  # max expected coins
        components.append(coins_normalized)
        
        # lost influence - already binary
        components.append(obs["lost_influence"].flatten())
        
        # eliminated players - already binary
        components.append(obs["is_eliminated"].astype(np.float32))
        
        # private hand - already binary
        components.append(obs["hand"].astype(np.float32))
        
        # game state - normalize discrete values
        components.append(np.array([obs["current_player"] / self.num_players]))
        components.append(np.array([obs["phase"] / 6.0]))  # 6 phases total
        components.append(np.array([obs["turn_count"] / 100.0]))  # normalize turn count
        
        # action context
        components.append(np.array([obs["pending_action"] / self.num_actions]))
        components.append(np.array([(obs["pending_target"] + 1) / (self.num_players + 1)]))  # +1 for -1 case
        components.append(np.array([obs["can_respond"]]))
        
        # action history - normalize
        hist_actions = obs["action_history"].astype(np.float32) / self.num_actions
        hist_players = obs["action_players"].astype(np.float32) / self.num_players
        hist_targets = (obs["action_targets"].astype(np.float32) + 1) / (self.num_players + 1)
        
        components.extend([hist_actions, hist_players, hist_targets])
        
        # concatenate all components
        flattened = np.concatenate(components, axis=0)
        
        return torch.FloatTensor(flattened)
    
    def get_valid_actions_mask(self, obs: Dict) -> torch.Tensor:
        """extract valid actions mask"""
        return torch.FloatTensor(obs["valid_actions"])


class CoupNetwork(nn.Module):
    """neural network for coup agent with shared backbone and separate heads"""
    
    def __init__(self, obs_size: int, num_actions: int, hidden_size: int = 128, lstm_size: int = 64):
        super().__init__()
        
        self.obs_size = obs_size
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.lstm_size = lstm_size
        
        # shared feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, lstm_size)
        )
        
        # lstm for memory (this is the key component for partial observability)
        self.lstm = nn.LSTM(lstm_size, lstm_size, batch_first=True)
        
        # policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(lstm_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_actions)
        )
        
        # value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, obs: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """forward pass through network"""
        batch_size = obs.size(0)
        
        # process observation through feature network
        features = self.feature_net(obs)
        
        # add sequence dimension for lstm (batch_size, seq_len=1, features)
        features = features.unsqueeze(1)
        
        # lstm processing
        lstm_out, new_hidden = self.lstm(features, hidden_state)
        
        # remove sequence dimension
        lstm_out = lstm_out.squeeze(1)
        
        # compute policy logits and value
        policy_logits = self.policy_head(lstm_out)
        value = self.value_head(lstm_out)
        
        return policy_logits, value, new_hidden
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """initialize hidden state for lstm"""
        h_0 = torch.zeros(1, batch_size, self.lstm_size)
        c_0 = torch.zeros(1, batch_size, self.lstm_size)
        return (h_0, c_0) 