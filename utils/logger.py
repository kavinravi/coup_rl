import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


@dataclass
class TrainingMetrics:
    """container for training metrics"""
    timestep: int
    episode: int
    
    # learning metrics
    policy_loss: float
    value_loss: float
    entropy: float
    kl_divergence: float
    learning_rate: float
    
    # game metrics
    episode_reward: float
    episode_length: int
    win_rate: float
    
    # performance metrics
    fps: float
    training_time: float
    
    # additional metrics
    custom_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.custom_metrics is None:
            self.custom_metrics = {}


@dataclass
class EvaluationMetrics:
    """container for evaluation metrics"""
    timestep: int
    
    # performance against baselines
    win_rates: Dict[str, float]
    avg_episode_length: float
    
    # elo rating
    elo_rating: float
    elo_change: float
    
    # detailed stats
    challenge_accuracy: float
    block_accuracy: float
    bluff_success_rate: float
    
    # game analysis
    action_distribution: Dict[str, float]
    avg_coins_per_game: float
    avg_eliminations: float


class MetricsBuffer:
    """buffer for collecting and aggregating metrics"""
    
    def __init__(self, maxlen: int = 1000):
        self.maxlen = maxlen
        self.reset()
    
    def reset(self):
        """reset all buffers"""
        self.rewards = deque(maxlen=self.maxlen)
        self.lengths = deque(maxlen=self.maxlen)
        self.policy_losses = deque(maxlen=self.maxlen)
        self.value_losses = deque(maxlen=self.maxlen)
        self.entropies = deque(maxlen=self.maxlen)
        self.kl_divergences = deque(maxlen=self.maxlen)
        self.custom_metrics = defaultdict(lambda: deque(maxlen=self.maxlen))
        
    def add_episode(self, reward: float, length: int, **kwargs):
        """add episode metrics"""
        self.rewards.append(reward)
        self.lengths.append(length)
        
        # add custom metrics
        for key, value in kwargs.items():
            self.custom_metrics[key].append(value)
    
    def add_training_step(self, policy_loss: float, value_loss: float, 
                         entropy: float, kl_div: float, **kwargs):
        """add training step metrics"""
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropies.append(entropy)
        self.kl_divergences.append(kl_div)
        
        # add custom metrics
        for key, value in kwargs.items():
            self.custom_metrics[key].append(value)
    
    def get_stats(self) -> Dict[str, float]:
        """get aggregated statistics"""
        stats = {}
        
        # episode stats
        if self.rewards:
            stats['reward_mean'] = np.mean(self.rewards)
            stats['reward_std'] = np.std(self.rewards)
            stats['reward_min'] = np.min(self.rewards)
            stats['reward_max'] = np.max(self.rewards)
            
        if self.lengths:
            stats['length_mean'] = np.mean(self.lengths)
            stats['length_std'] = np.std(self.lengths)
            
        # training stats
        if self.policy_losses:
            stats['policy_loss_mean'] = np.mean(self.policy_losses)
            stats['value_loss_mean'] = np.mean(self.value_losses)
            stats['entropy_mean'] = np.mean(self.entropies)
            stats['kl_div_mean'] = np.mean(self.kl_divergences)
            
        # custom metrics
        for key, values in self.custom_metrics.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                
        return stats


class CoupLogger:
    """comprehensive logger for coup rl training"""
    
    def __init__(self, log_dir: str, config: Dict[str, Any]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.start_time = time.time()
        
        # create subdirectories
        self.tensorboard_dir = self.log_dir / "tensorboard"
        self.checkpoints_dir = self.log_dir / "checkpoints"
        self.logs_dir = self.log_dir / "logs"
        
        for dir_path in [self.tensorboard_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # setup logging
        self._setup_logging()
        
        # setup tensorboard
        if config.get('logging', {}).get('tensorboard', False):
            self.tb_writer = SummaryWriter(str(self.tensorboard_dir))
        else:
            self.tb_writer = None
            
        # setup wandb
        self.wandb_enabled = config.get('logging', {}).get('wandb', False)
        if self.wandb_enabled:
            self._setup_wandb()
            
        # metrics buffer
        self.metrics_buffer = MetricsBuffer()
        
        # training tracking
        self.total_timesteps = 0
        self.total_episodes = 0
        self.best_performance = {}
        
        # save config
        self._save_config()
        
        self.logger.info(f"Logger initialized: {self.log_dir}")
    
    def _setup_logging(self):
        """setup file and console logging"""
        # create logger
        self.logger = logging.getLogger('coup_training')
        self.logger.setLevel(logging.DEBUG)
        
        # clear existing handlers
        self.logger.handlers.clear()
        
        # console handler
        console_handler = logging.StreamHandler()
        console_level = getattr(logging, self.config.get('logging', {}).get('console_level', 'INFO'))
        console_handler.setLevel(console_level)
        
        # file handler
        file_handler = logging.FileHandler(self.logs_dir / 'training.log')
        file_level = getattr(logging, self.config.get('logging', {}).get('file_level', 'DEBUG'))
        file_handler.setLevel(file_level)
        
        # formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def _setup_wandb(self):
        """setup weights and biases logging"""
        try:
            import wandb
            
            wandb_config = self.config.get('logging', {})
            wandb.init(
                project=wandb_config.get('wandb_project', 'coup-rl'),
                entity=wandb_config.get('wandb_entity'),
                tags=wandb_config.get('wandb_tags', []),
                config=self.config,
                dir=str(self.log_dir)
            )
            
            self.wandb = wandb
            self.logger.info("Weights & Biases logging enabled")
            
        except ImportError:
            self.logger.warning("wandb not installed, disabling cloud logging")
            self.wandb_enabled = False
            self.wandb = None
    
    def _save_config(self):
        """save configuration to file"""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
    
    def log_training_step(self, timestep: int, metrics: Dict[str, float]):
        """log training step metrics"""
        self.total_timesteps = timestep
        
        # add to buffer
        self.metrics_buffer.add_training_step(
            policy_loss=metrics.get('policy_loss', 0),
            value_loss=metrics.get('value_loss', 0),
            entropy=metrics.get('entropy', 0),
            kl_div=metrics.get('kl_divergence', 0),
            **{k: v for k, v in metrics.items() if k not in ['policy_loss', 'value_loss', 'entropy', 'kl_divergence']}
        )
        
        # log to tensorboard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(f'training/{key}', value, timestep)
        
        # log to wandb
        if self.wandb_enabled and self.wandb:
            wandb_metrics = {f'train/{k}': v for k, v in metrics.items()}
            self.wandb.log(wandb_metrics, step=timestep)
    
    def log_episode(self, episode: int, reward: float, length: int, **kwargs):
        """log episode completion"""
        self.total_episodes = episode
        
        # add to buffer
        self.metrics_buffer.add_episode(reward, length, **kwargs)
        
        # log to tensorboard
        if self.tb_writer:
            self.tb_writer.add_scalar('episode/reward', reward, episode)
            self.tb_writer.add_scalar('episode/length', length, episode)
            for key, value in kwargs.items():
                self.tb_writer.add_scalar(f'episode/{key}', value, episode)
        
        # log to wandb
        if self.wandb_enabled and self.wandb:
            wandb_metrics = {
                'episode/reward': reward,
                'episode/length': length,
                **{f'episode/{k}': v for k, v in kwargs.items()}
            }
            self.wandb.log(wandb_metrics, step=self.total_timesteps)
    
    def log_evaluation(self, timestep: int, eval_metrics: EvaluationMetrics):
        """log evaluation results"""
        self.logger.info(f"Evaluation at step {timestep}:")
        self.logger.info(f"  Elo Rating: {eval_metrics.elo_rating:.1f} ({eval_metrics.elo_change:+.1f})")
        self.logger.info(f"  Win rates: {eval_metrics.win_rates}")
        self.logger.info(f"  Challenge accuracy: {eval_metrics.challenge_accuracy:.3f}")
        self.logger.info(f"  Block accuracy: {eval_metrics.block_accuracy:.3f}")
        self.logger.info(f"  Bluff success rate: {eval_metrics.bluff_success_rate:.3f}")
        
        # update best performance
        if 'elo_rating' not in self.best_performance or eval_metrics.elo_rating > self.best_performance['elo_rating']:
            self.best_performance.update(asdict(eval_metrics))
            self.logger.info(f"New best performance! Elo: {eval_metrics.elo_rating:.1f}")
        
        # log to tensorboard
        if self.tb_writer:
            self.tb_writer.add_scalar('eval/elo_rating', eval_metrics.elo_rating, timestep)
            self.tb_writer.add_scalar('eval/elo_change', eval_metrics.elo_change, timestep)
            self.tb_writer.add_scalar('eval/avg_episode_length', eval_metrics.avg_episode_length, timestep)
            self.tb_writer.add_scalar('eval/challenge_accuracy', eval_metrics.challenge_accuracy, timestep)
            self.tb_writer.add_scalar('eval/block_accuracy', eval_metrics.block_accuracy, timestep)
            self.tb_writer.add_scalar('eval/bluff_success_rate', eval_metrics.bluff_success_rate, timestep)
            
            for opponent, win_rate in eval_metrics.win_rates.items():
                self.tb_writer.add_scalar(f'eval/win_rate_{opponent}', win_rate, timestep)
        
        # log to wandb
        if self.wandb_enabled and self.wandb:
            wandb_metrics = {
                'eval/elo_rating': eval_metrics.elo_rating,
                'eval/elo_change': eval_metrics.elo_change,
                'eval/avg_episode_length': eval_metrics.avg_episode_length,
                'eval/challenge_accuracy': eval_metrics.challenge_accuracy,
                'eval/block_accuracy': eval_metrics.block_accuracy,
                'eval/bluff_success_rate': eval_metrics.bluff_success_rate,
                **{f'eval/win_rate_{k}': v for k, v in eval_metrics.win_rates.items()}
            }
            self.wandb.log(wandb_metrics, step=timestep)
    
    def log_checkpoint(self, timestep: int, checkpoint_path: str, is_best: bool = False):
        """log checkpoint save"""
        self.logger.info(f"Checkpoint saved at step {timestep}: {checkpoint_path}")
        if is_best:
            self.logger.info("This is the best checkpoint so far!")
    
    def log_stats(self, timestep: int):
        """log aggregated statistics"""
        stats = self.metrics_buffer.get_stats()
        
        if stats:
            self.logger.info(f"Stats at step {timestep}:")
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            # log to tensorboard
            if self.tb_writer:
                for key, value in stats.items():
                    self.tb_writer.add_scalar(f'stats/{key}', value, timestep)
            
            # log to wandb
            if self.wandb_enabled and self.wandb:
                wandb_metrics = {f'stats/{k}': v for k, v in stats.items()}
                self.wandb.log(wandb_metrics, step=timestep)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """get training summary"""
        total_time = time.time() - self.start_time
        
        return {
            'total_timesteps': self.total_timesteps,
            'total_episodes': self.total_episodes,
            'total_time': total_time,
            'timesteps_per_second': self.total_timesteps / total_time if total_time > 0 else 0,
            'episodes_per_second': self.total_episodes / total_time if total_time > 0 else 0,
            'best_performance': self.best_performance,
            'final_stats': self.metrics_buffer.get_stats()
        }
    
    def close(self):
        """close logger and cleanup"""
        # final summary
        summary = self.get_training_summary()
        self.logger.info("Training completed!")
        self.logger.info(f"Total timesteps: {summary['total_timesteps']}")
        self.logger.info(f"Total episodes: {summary['total_episodes']}")
        self.logger.info(f"Total time: {summary['total_time']:.2f}s")
        self.logger.info(f"Best Elo: {summary['best_performance'].get('elo_rating', 'N/A')}")
        
        # save summary
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # close tensorboard
        if self.tb_writer:
            self.tb_writer.close()
        
        # close wandb
        if self.wandb_enabled and self.wandb:
            self.wandb.finish()
        
        self.logger.info(f"Training summary saved: {summary_path}") 