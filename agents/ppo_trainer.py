import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from coup_agent import CoupAgent
import sys
sys.path.append('../envs')


@dataclass
class PPOConfig:
    """configuration for ppo training"""
    # training hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount factor
    gae_lambda: float = 0.95  # gae parameter
    clip_range: float = 0.2  # ppo clip range
    entropy_coef: float = 0.01  # entropy regularization
    value_coef: float = 0.5  # value loss coefficient
    max_grad_norm: float = 0.5  # gradient clipping
    
    # training schedule
    ppo_epochs: int = 4  # number of ppo epochs per update
    batch_size: int = 64  # minibatch size
    n_steps: int = 128  # rollout length
    
    # regularization
    target_kl: float = 0.01  # target kl divergence for early stopping
    use_gae: bool = True  # use generalized advantage estimation


class ExperienceBuffer:
    """buffer for storing experience during rollouts"""
    
    def __init__(self, n_steps: int, n_envs: int, obs_size: int, device: str = "cpu"):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_size = obs_size
        self.device = device
        
        # storage buffers
        self.observations = torch.zeros(n_steps, n_envs, obs_size, device=device)
        self.actions = torch.zeros(n_steps, n_envs, dtype=torch.long, device=device)
        self.rewards = torch.zeros(n_steps, n_envs, device=device)
        self.dones = torch.zeros(n_steps, n_envs, dtype=torch.bool, device=device)
        self.values = torch.zeros(n_steps, n_envs, device=device)
        self.log_probs = torch.zeros(n_steps, n_envs, device=device)
        
        # for lstm hidden states
        self.hidden_states = []
        
        # tracking
        self.step = 0
        self.full = False
    
    def add(self, obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, 
            done: torch.Tensor, value: torch.Tensor, log_prob: torch.Tensor,
            hidden_state: Optional[Tuple] = None):
        """add experience to buffer"""
        self.observations[self.step] = obs
        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.log_probs[self.step] = log_prob
        
        if hidden_state is not None:
            self.hidden_states.append(hidden_state)
        
        self.step = (self.step + 1) % self.n_steps
        if self.step == 0:
            self.full = True
    
    def get_batch(self, start_idx: int, end_idx: int) -> Dict[str, torch.Tensor]:
        """get a batch of experiences"""
        return {
            'observations': self.observations[start_idx:end_idx],
            'actions': self.actions[start_idx:end_idx],
            'rewards': self.rewards[start_idx:end_idx],
            'dones': self.dones[start_idx:end_idx],
            'values': self.values[start_idx:end_idx],
            'log_probs': self.log_probs[start_idx:end_idx]
        }
    
    def clear(self):
        """clear the buffer"""
        self.step = 0
        self.full = False
        self.hidden_states = []


class PPOTrainer:
    """ppo trainer for coup agents with lstm support"""
    
    def __init__(self, agent: CoupAgent, config: PPOConfig = None):
        self.agent = agent
        self.config = config or PPOConfig()
        
        # optimizer
        self.optimizer = torch.optim.Adam(agent.network.parameters(), lr=self.config.learning_rate)
        
        # experience buffer
        self.buffer = ExperienceBuffer(
            n_steps=self.config.n_steps,
            n_envs=1,  # single environment for now
            obs_size=agent.obs_processor.obs_size,
            device=agent.device
        )
        
        # training statistics
        self.stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'total_loss': [],
            'grad_norm': []
        }
    
    def collect_experience(self, env, n_steps: int) -> Dict[str, float]:
        """collect experience from environment"""
        self.agent.set_training_mode(False)  # evaluation mode during collection
        
        # reset environment and agent
        obs, info = env.reset()
        self.agent.reset_hidden_state()
        
        episode_rewards = []
        episode_lengths = []
        current_episode_reward = 0
        current_episode_length = 0
        
        for step in range(n_steps):
            # get action from agent
            action, action_info = self.agent.get_action(obs)
            
            # step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # store experience
            obs_tensor = self.agent.obs_processor.process_observation(obs)
            self.buffer.add(
                obs=obs_tensor,
                action=torch.tensor(action, dtype=torch.long),
                reward=torch.tensor(reward, dtype=torch.float),
                done=torch.tensor(done, dtype=torch.bool),
                value=torch.tensor(action_info['value'], dtype=torch.float),
                log_prob=torch.tensor(np.log(action_info['selected_prob']), dtype=torch.float)
            )
            
            # update tracking
            current_episode_reward += reward
            current_episode_length += 1
            
            # handle episode end
            if done:
                episode_rewards.append(current_episode_reward)
                episode_lengths.append(current_episode_length)
                
                # reset for next episode
                obs, info = env.reset()
                self.agent.reset_hidden_state()
                current_episode_reward = 0
                current_episode_length = 0
            else:
                obs = next_obs
        
        # return statistics
        return {
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'mean_length': np.mean(episode_lengths) if episode_lengths else 0.0,
            'num_episodes': len(episode_rewards)
        }
    
    def compute_gae(self, next_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """compute generalized advantage estimation"""
        advantages = torch.zeros_like(self.buffer.rewards)
        returns = torch.zeros_like(self.buffer.rewards)
        
        # bootstrap from next value
        last_advantage = 0.0
        
        # compute backwards through time
        for t in reversed(range(self.config.n_steps)):
            if t == self.config.n_steps - 1:
                # last step
                next_non_terminal = 1.0 - self.buffer.dones[t].float()
                next_value_t = next_value
            else:
                # intermediate step
                next_non_terminal = 1.0 - self.buffer.dones[t].float()
                next_value_t = self.buffer.values[t + 1]
            
            # td error
            delta = self.buffer.rewards[t] + self.config.gamma * next_value_t * next_non_terminal - self.buffer.values[t]
            
            # gae
            advantages[t] = last_advantage = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * last_advantage
            
            # returns
            returns[t] = advantages[t] + self.buffer.values[t]
        
        return advantages, returns
    
    def update_policy(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """update policy using ppo"""
        self.agent.set_training_mode(True)
        
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # flatten data for training
        obs_flat = self.buffer.observations.view(-1, self.agent.obs_processor.obs_size)
        actions_flat = self.buffer.actions.view(-1)
        old_log_probs_flat = self.buffer.log_probs.view(-1)
        advantages_flat = advantages.view(-1)
        returns_flat = returns.view(-1)
        
        # training statistics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_div = 0
        n_updates = 0
        
        # ppo epochs
        for epoch in range(self.config.ppo_epochs):
            # create random batches
            indices = torch.randperm(obs_flat.size(0))
            
            for start_idx in range(0, obs_flat.size(0), self.config.batch_size):
                end_idx = start_idx + self.config.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # get batch
                batch_obs = obs_flat[batch_indices]
                batch_actions = actions_flat[batch_indices]
                batch_old_log_probs = old_log_probs_flat[batch_indices]
                batch_advantages = advantages_flat[batch_indices]
                batch_returns = returns_flat[batch_indices]
                
                # forward pass
                policy_logits, values, _ = self.agent.network(batch_obs)
                
                # compute new log probabilities
                log_probs = F.log_softmax(policy_logits, dim=-1)
                new_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                
                # compute ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # value loss
                value_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # entropy loss
                entropy = -(F.softmax(policy_logits, dim=-1) * log_probs).sum(dim=-1).mean()
                
                # total loss
                total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
                
                # backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                
                # compute kl divergence for early stopping
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    
                    # early stopping if kl divergence too large
                    if kl_div > self.config.target_kl:
                        break
                
                # update statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_div += kl_div.item()
                n_updates += 1
        
        # clear buffer
        self.buffer.clear()
        
        # return training statistics
        return {
            'policy_loss': total_policy_loss / n_updates if n_updates > 0 else 0.0,
            'value_loss': total_value_loss / n_updates if n_updates > 0 else 0.0,
            'entropy': total_entropy / n_updates if n_updates > 0 else 0.0,
            'kl_divergence': total_kl_div / n_updates if n_updates > 0 else 0.0,
            'n_updates': n_updates
        }
    
    def train_step(self, env) -> Dict[str, float]:
        """single training step"""
        # collect experience
        rollout_stats = self.collect_experience(env, self.config.n_steps)
        
        # compute advantages
        advantages, returns = self.compute_gae()
        
        # update policy
        update_stats = self.update_policy(advantages, returns)
        
        # combine statistics
        stats = {**rollout_stats, **update_stats}
        
        # update internal statistics
        for key, value in update_stats.items():
            if key in self.stats:
                self.stats[key].append(value)
        
        return stats
    
    def save_checkpoint(self, path: str):
        """save training checkpoint"""
        torch.save({
            'agent_state_dict': self.agent.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stats': self.stats
        }, path)
    
    def load_checkpoint(self, path: str):
        """load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.agent.device)
        self.agent.network.load_state_dict(checkpoint['agent_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.stats = checkpoint['stats'] 