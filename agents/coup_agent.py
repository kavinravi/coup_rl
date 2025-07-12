import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from coup_network import CoupNetwork, ObservationProcessor
from coup_game import Action
import sys
sys.path.append('../envs')


class CoupAgent:
    """ppo agent for coup game with lstm memory"""
    
    def __init__(self, num_players: int = 2, device: str = "cpu"):
        self.num_players = num_players
        self.device = device
        
        # initialize observation processor
        self.obs_processor = ObservationProcessor(num_players)
        
        # initialize network
        self.network = CoupNetwork(
            obs_size=self.obs_processor.obs_size,
            num_actions=len(Action)
        ).to(device)
        
        # hidden state for lstm (will be reset at episode start)
        self.hidden_state = None
        
        # tracking for training
        self.training_mode = True
        
    def reset_hidden_state(self):
        """reset lstm hidden state (call at episode start)"""
        self.hidden_state = None
    
    def set_training_mode(self, training: bool):
        """set training mode"""
        self.training_mode = training
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def get_action(self, obs: Dict, deterministic: bool = False) -> Tuple[int, Dict]:
        """get action from observation"""
        with torch.no_grad():
            # process observation
            obs_tensor = self.obs_processor.process_observation(obs).unsqueeze(0).to(self.device)
            valid_actions_mask = self.obs_processor.get_valid_actions_mask(obs).to(self.device)
            
            # forward pass through network
            policy_logits, value, new_hidden = self.network(obs_tensor, self.hidden_state)
            
            # update hidden state
            self.hidden_state = new_hidden
            
            # apply action masking (set invalid actions to very negative values)
            masked_logits = policy_logits.clone()
            masked_logits[0, valid_actions_mask == 0] = -1e8
            
            # compute action probabilities
            action_probs = F.softmax(masked_logits, dim=-1)
            
            # sample action
            if deterministic:
                # greedy action selection
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # stochastic sampling
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
            
            # return action and additional info
            action_info = {
                'action_probs': action_probs.cpu().numpy(),
                'value': value.item(),
                'valid_actions': valid_actions_mask.cpu().numpy(),
                'selected_prob': action_probs[0, action].item()
            }
            
            return action, action_info
    
    def evaluate_actions(self, obs_batch: torch.Tensor, actions: torch.Tensor, 
                        hidden_states: Optional[List] = None) -> Dict[str, torch.Tensor]:
        """evaluate actions for training (used by ppo)"""
        batch_size = obs_batch.size(0)
        
        # if no hidden states provided, use zero initialization
        if hidden_states is None:
            hidden_states = [self.network.init_hidden(batch_size)]
        
        # process all timesteps
        all_logits = []
        all_values = []
        
        for t in range(len(hidden_states)):
            # get observations for this timestep
            if t < obs_batch.size(1):  # check if we have observations for this timestep
                obs_t = obs_batch[:, t]
            else:
                break
            
            # forward pass
            logits, values, new_hidden = self.network(obs_t, hidden_states[t])
            
            all_logits.append(logits)
            all_values.append(values)
            
            # update hidden state for next timestep
            if t + 1 < len(hidden_states):
                hidden_states[t + 1] = new_hidden
        
        # stack results
        policy_logits = torch.stack(all_logits, dim=1)  # (batch, time, actions)
        values = torch.stack(all_values, dim=1)        # (batch, time, 1)
        
        # compute action probabilities
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # compute log probabilities for selected actions
        log_probs = F.log_softmax(policy_logits, dim=-1)
        selected_log_probs = log_probs.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        
        # compute entropy for exploration bonus
        entropy = -(action_probs * log_probs).sum(dim=-1)
        
        return {
            'log_probs': selected_log_probs,
            'values': values.squeeze(-1),
            'entropy': entropy,
            'action_probs': action_probs
        }
    
    def get_value(self, obs: Dict) -> float:
        """get value estimate for current state"""
        with torch.no_grad():
            obs_tensor = self.obs_processor.process_observation(obs).unsqueeze(0).to(self.device)
            _, value, _ = self.network(obs_tensor, self.hidden_state)
            return value.item()
    
    def save_model(self, path: str):
        """save model weights"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'num_players': self.num_players,
            'obs_size': self.obs_processor.obs_size
        }, path)
    
    def load_model(self, path: str):
        """load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        
        # verify compatibility
        assert checkpoint['num_players'] == self.num_players
        assert checkpoint['obs_size'] == self.obs_processor.obs_size


class ActionMasker:
    """utility class for handling action masking in different scenarios"""
    
    @staticmethod
    def apply_mask(logits: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e8) -> torch.Tensor:
        """apply action mask to logits"""
        masked_logits = logits.clone()
        masked_logits[mask == 0] = mask_value
        return masked_logits
    
    @staticmethod
    def get_valid_action_indices(mask: torch.Tensor) -> List[int]:
        """get indices of valid actions"""
        return torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
    
    @staticmethod
    def normalize_probabilities(probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """normalize probabilities to sum to 1 over valid actions only"""
        masked_probs = probs * mask
        return masked_probs / masked_probs.sum(dim=-1, keepdim=True)


# utility functions for creating and managing agents
def create_coup_agent(num_players: int = 2, device: str = "cpu") -> CoupAgent:
    """create a coup agent with default settings"""
    return CoupAgent(num_players=num_players, device=device)


def create_agent_pair(num_players: int = 2, device: str = "cpu") -> Tuple[CoupAgent, CoupAgent]:
    """create a pair of agents for self-play"""
    agent1 = create_coup_agent(num_players, device)
    agent2 = create_coup_agent(num_players, device)
    return agent1, agent2 