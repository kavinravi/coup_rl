import os
import sys
import yaml
import argparse
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
import time

# add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

from envs.coup_env import CoupEnv
from agents.coup_agent import CoupAgent
from agents.ppo_trainer import PPOTrainer, PPOConfig
from eval.baseline_agents import create_baseline_agent
from eval.match_runner import MatchRunner
from eval.elo_system import EloRatingSystem
from utils.logger import CoupLogger, EvaluationMetrics
from utils.metrics import GameStatistics, EvaluationMetrics as EvalMetrics
from utils.checkpoint_manager import CheckpointManager


class SelfPlayTrainer:
    """self-play trainer for coup agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['agent']['device']
        
        # set random seeds
        self._set_random_seeds()
        
        # create environment
        self.env = CoupEnv(
            num_players=config['environment']['num_players'],
            max_turns=config['environment']['max_turns'],
            render_mode=config['environment']['render_mode']
        )
        
        # create agent
        self.agent = CoupAgent(
            num_players=config['environment']['num_players'],
            device=self.device
        )
        
        # create PPO trainer
        ppo_config = PPOConfig(**config['ppo'])
        self.trainer = PPOTrainer(self.agent, ppo_config)
        
        # create opponent pool for self-play
        self.opponent_pool = []
        self.opponent_selection_strategy = config['self_play']['selection_strategy']
        
        # create evaluation system
        self.eval_env = CoupEnv(
            num_players=config['environment']['num_players'],
            max_turns=config['environment']['max_turns']
        )
        self.match_runner = MatchRunner(
            num_players=config['environment']['num_players'],
            max_steps=config['environment']['max_turns'],
            verbose=False
        )
        self.elo_system = EloRatingSystem()
        
        # create baseline agents for evaluation
        self.baseline_agents = {}
        for agent_name in config['evaluation']['eval_opponents']:
            self.baseline_agents[agent_name] = create_baseline_agent(agent_name)
        
        # create logger
        log_dir = f"logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger = CoupLogger(log_dir, config)
        
        # create checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=os.path.join(log_dir, "checkpoints"),
            config=config
        )
        
        # training state
        self.total_timesteps = 0
        self.total_episodes = 0
        self.best_elo_rating = 0
        self.no_improvement_count = 0
        
        # reward shaping
        self.reward_config = config['rewards']
        
        # early stopping
        self.early_stopping_config = config['training']['early_stopping']
        
        self.logger.logger.info("Self-play trainer initialized")
        self.logger.logger.info(f"Device: {self.device}")
        self.logger.logger.info(f"Environment: {config['environment']['num_players']} players")
        self.logger.logger.info(f"Total timesteps: {config['training']['total_timesteps']}")
    
    def _set_random_seeds(self):
        """set random seeds for reproducibility"""
        seed = self.config.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _create_self_play_environment(self) -> CoupEnv:
        """create environment for self-play training"""
        return CoupEnv(
            num_players=self.config['environment']['num_players'],
            max_turns=self.config['environment']['max_turns']
        )
    
    def _get_opponent_agent(self) -> CoupAgent:
        """get opponent agent for self-play"""
        if not self.opponent_pool:
            # use current agent as opponent if no pool exists
            return self.agent
        
        if self.opponent_selection_strategy == "recent":
            # select from recent opponents
            return self.opponent_pool[-1]
        elif self.opponent_selection_strategy == "random":
            # select random opponent
            return random.choice(self.opponent_pool)
        elif self.opponent_selection_strategy == "tournament":
            # select based on tournament (not implemented yet)
            return random.choice(self.opponent_pool)
        else:
            return self.opponent_pool[-1]
    
    def _update_opponent_pool(self):
        """update opponent pool with current agent"""
        if not self.config['self_play']['enabled']:
            return
        
        # create copy of current agent
        opponent = CoupAgent(
            num_players=self.config['environment']['num_players'],
            device=self.device
        )
        
        # copy weights
        opponent.network.load_state_dict(self.agent.network.state_dict())
        
        # add to pool
        self.opponent_pool.append(opponent)
        
        # keep pool size manageable
        max_pool_size = self.config['self_play']['opponent_pool_size']
        if len(self.opponent_pool) > max_pool_size:
            self.opponent_pool = self.opponent_pool[-max_pool_size:]
        
        self.logger.logger.info(f"Updated opponent pool (size: {len(self.opponent_pool)})")
    
    def _shape_reward(self, reward: float, info: Dict[str, Any]) -> float:
        """apply reward shaping"""
        shaped_reward = reward
        
        # terminal rewards
        if info.get('game_over', False):
            if info.get('winner') == 0:  # assuming agent is player 0
                shaped_reward = self.reward_config['win_reward']
            else:
                shaped_reward = self.reward_config['loss_reward']
        
        # intermediate rewards
        if 'challenge_success' in info:
            shaped_reward += self.reward_config['challenge_success']
        
        if 'challenge_failure' in info:
            shaped_reward += self.reward_config['challenge_failure']
        
        if 'successful_bluff' in info:
            shaped_reward += self.reward_config['successful_bluff']
        
        if 'failed_bluff' in info:
            shaped_reward += self.reward_config['failed_bluff']
        
        if 'coins_gained' in info:
            shaped_reward += info['coins_gained'] * self.reward_config['gain_coins']
        
        if 'coins_lost' in info:
            shaped_reward += info['coins_lost'] * self.reward_config['lose_coins']
        
        if 'eliminate_opponent' in info:
            shaped_reward += self.reward_config['eliminate_opponent']
        
        if 'lose_influence' in info:
            shaped_reward += self.reward_config['lose_influence']
        
        # exploration and efficiency bonuses
        shaped_reward += self.reward_config['turn_penalty']  # small penalty per turn
        
        return shaped_reward
    
    def train_step(self) -> Dict[str, Any]:
        """perform one training step"""
        # collect experience and update policy
        stats = self.trainer.train_step(self.env)
        
        # update total timesteps
        self.total_timesteps += self.trainer.config.n_steps
        
        return stats
    
    def evaluate_agent(self, num_episodes: int = None) -> EvaluationMetrics:
        """evaluate agent against baselines"""
        if num_episodes is None:
            num_episodes = self.config['evaluation']['num_eval_episodes']
        
        self.logger.logger.info(f"Evaluating agent over {num_episodes} episodes")
        
        # results storage
        win_rates = {}
        all_results = []
        
        # evaluate against each baseline
        for baseline_name, baseline_agent in self.baseline_agents.items():
            wins = 0
            total_episodes = 0
            episode_lengths = []
            
            for episode in range(num_episodes):
                # create agents list (agent always at index 0)
                agents = [self.agent, baseline_agent]
                
                # run match
                result = self.match_runner.run_match(agents)
                all_results.append(result)
                
                # track statistics
                total_episodes += 1
                episode_lengths.append(result.game_length)
                
                if result.winner == 0:  # agent won
                    wins += 1
                
                # update elo rating
                if self.config['evaluation']['elo_update']:
                    # get current ratings
                    agent_stats = self.elo_system.get_player("agent")
                    baseline_stats = self.elo_system.get_player(baseline_name)
                    
                    # update ratings
                    if result.winner == 0:
                        self.elo_system.update_ratings("agent", baseline_name, 1.0)
                    else:
                        self.elo_system.update_ratings("agent", baseline_name, 0.0)
            
            # calculate win rate
            win_rate = wins / total_episodes if total_episodes > 0 else 0
            win_rates[baseline_name] = win_rate
            
            self.logger.logger.info(f"Win rate vs {baseline_name}: {win_rate:.3f}")
        
        # get current elo rating
        agent_stats = self.elo_system.get_player("agent")
        current_elo = agent_stats.current_rating
        elo_change = current_elo - self.best_elo_rating
        
        # compute detailed statistics
        eval_metrics = EvalMetrics()
        game_results = []
        
        for result in all_results:
            # convert match result to game result format
            game_result = {
                'length': result.game_length,
                'winner': result.winner,
                'players': result.player_stats,
                'actions': [],  # would need to be collected during match
                'challenges': [],  # would need to be collected during match
                'blocks': []  # would need to be collected during match
            }
            game_results.append(game_result)
        
        # simplified stats calculation since we don't have all the detailed data
        detailed_stats = {
            'challenge_accuracy': 0.0,
            'block_accuracy': 0.0,
            'bluff_success_rate': 0.0,
            'action_distribution': {},
            'avg_eliminations': 0.0
        }
        
        # compute what we can from available data
        if all_results:
            # compute basic stats from player stats
            total_challenges = sum(sum(r.player_stats.get(i, {}).get('challenges_made', 0) for i in range(len(r.player_names))) for r in all_results)
            total_challenge_wins = sum(sum(r.player_stats.get(i, {}).get('challenges_won', 0) for i in range(len(r.player_names))) for r in all_results)
            
            if total_challenges > 0:
                detailed_stats['challenge_accuracy'] = total_challenge_wins / total_challenges
            
            total_blocks = sum(sum(r.player_stats.get(i, {}).get('blocks_made', 0) for i in range(len(r.player_names))) for r in all_results)
            total_block_success = sum(sum(r.player_stats.get(i, {}).get('blocks_successful', 0) for i in range(len(r.player_names))) for r in all_results)
            
            if total_blocks > 0:
                detailed_stats['block_accuracy'] = total_block_success / total_blocks
            
            total_bluffs = sum(sum(r.player_stats.get(i, {}).get('bluffs_attempted', 0) for i in range(len(r.player_names))) for r in all_results)
            total_bluff_success = sum(sum(r.player_stats.get(i, {}).get('bluffs_successful', 0) for i in range(len(r.player_names))) for r in all_results)
            
            if total_bluffs > 0:
                detailed_stats['bluff_success_rate'] = total_bluff_success / total_bluffs
        
        # create evaluation metrics
        evaluation_metrics = EvaluationMetrics(
            timestep=self.total_timesteps,
            win_rates=win_rates,
            avg_episode_length=np.mean([r.game_length for r in all_results]),
            elo_rating=current_elo,
            elo_change=elo_change,
            challenge_accuracy=detailed_stats.get('challenge_accuracy', 0),
            block_accuracy=detailed_stats.get('block_accuracy', 0),
            bluff_success_rate=detailed_stats.get('bluff_success_rate', 0),
            action_distribution=detailed_stats.get('action_distribution', {}),
            avg_coins_per_game=0,  # would need to be computed from game state
            avg_eliminations=detailed_stats.get('avg_eliminations', 0)
        )
        
        return evaluation_metrics
    
    def should_stop_training(self, eval_metrics: EvaluationMetrics) -> bool:
        """check if training should stop early"""
        if not self.early_stopping_config['enabled']:
            return False
        
        # check if performance improved
        if eval_metrics.elo_rating > self.best_elo_rating + self.early_stopping_config['min_delta']:
            self.best_elo_rating = eval_metrics.elo_rating
            self.no_improvement_count = 0
            return False
        
        # increment no improvement counter
        self.no_improvement_count += 1
        
        # check if we should stop
        if self.no_improvement_count >= self.early_stopping_config['patience']:
            self.logger.logger.info(f"Early stopping: no improvement for {self.no_improvement_count} evaluations")
            return True
        
        return False
    
    def train(self):
        """main training loop"""
        self.logger.logger.info("Starting training")
        
        # training configuration
        total_timesteps = self.config['training']['total_timesteps']
        eval_frequency = self.config['training']['eval_frequency']
        checkpoint_frequency = self.config['training']['checkpoint_frequency']
        log_frequency = self.config['training']['log_frequency']
        
        # main training loop
        while self.total_timesteps < total_timesteps:
            # training step
            start_time = time.time()
            train_stats = self.train_step()
            step_time = time.time() - start_time
            
            # log training step
            if self.total_timesteps % log_frequency == 0:
                train_stats['fps'] = self.trainer.config.n_steps / step_time
                self.logger.log_training_step(self.total_timesteps, train_stats)
            
            # evaluation
            if self.total_timesteps % eval_frequency == 0:
                eval_metrics = self.evaluate_agent()
                self.logger.log_evaluation(self.total_timesteps, eval_metrics)
                
                # check early stopping
                if self.should_stop_training(eval_metrics):
                    break
            
            # checkpointing
            if self.total_timesteps % checkpoint_frequency == 0:
                # create checkpoint metrics
                checkpoint_metrics = {
                    'timestep': self.total_timesteps,
                    'elo_rating': self.best_elo_rating,
                    'training_stats': train_stats
                }
                
                # save checkpoint
                is_best = self.best_elo_rating == eval_metrics.elo_rating if 'eval_metrics' in locals() else False
                checkpoint_path = self.checkpoint_manager.save_checkpoint(
                    timestep=self.total_timesteps,
                    agent=self.agent,
                    trainer=self.trainer,
                    metrics=checkpoint_metrics,
                    is_best=is_best
                )
                
                self.logger.log_checkpoint(self.total_timesteps, checkpoint_path, is_best)
            
            # update opponent pool
            if (self.config['self_play']['enabled'] and 
                self.total_timesteps % self.config['self_play']['update_frequency'] == 0):
                self._update_opponent_pool()
            
            # log aggregated statistics
            if self.total_timesteps % (log_frequency * 10) == 0:
                self.logger.log_stats(self.total_timesteps)
        
        # final evaluation
        self.logger.logger.info("Training completed, running final evaluation")
        final_eval_metrics = self.evaluate_agent(num_episodes=self.config['evaluation']['tournament_size'])
        self.logger.log_evaluation(self.total_timesteps, final_eval_metrics)
        
        # save final checkpoint
        final_checkpoint_metrics = {
            'timestep': self.total_timesteps,
            'elo_rating': final_eval_metrics.elo_rating,
            'final_evaluation': True
        }
        
        final_checkpoint_path = self.checkpoint_manager.save_checkpoint(
            timestep=self.total_timesteps,
            agent=self.agent,
            trainer=self.trainer,
            metrics=final_checkpoint_metrics,
            is_best=True,
            checkpoint_name="final_checkpoint"
        )
        
        self.logger.log_checkpoint(self.total_timesteps, final_checkpoint_path, True)
        
        # training summary
        self.logger.logger.info("Training Summary:")
        self.logger.logger.info(f"Total timesteps: {self.total_timesteps}")
        self.logger.logger.info(f"Total episodes: {self.total_episodes}")
        self.logger.logger.info(f"Best Elo rating: {self.best_elo_rating}")
        self.logger.logger.info(f"Final Elo rating: {final_eval_metrics.elo_rating}")
        
        # close logger
        self.logger.close()


def load_config(config_path: str) -> Dict[str, Any]:
    """load training configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """main training function"""
    parser = argparse.ArgumentParser(description="Train Coup RL Agent")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cpu/cuda)")
    parser.add_argument("--num_players", type=int, default=None,
                       help="Number of players")
    parser.add_argument("--total_timesteps", type=int, default=None,
                       help="Total training timesteps")
    
    args = parser.parse_args()
    
    # load configuration
    config = load_config(args.config)
    
    # override config with command line arguments
    if args.device:
        config['agent']['device'] = args.device
    if args.num_players:
        config['environment']['num_players'] = args.num_players
    if args.total_timesteps:
        config['training']['total_timesteps'] = args.total_timesteps
    
    # create trainer
    trainer = SelfPlayTrainer(config)
    
    # resume from checkpoint if specified
    if args.resume:
        trainer.logger.logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
            args.resume, trainer.agent, trainer.trainer
        )
        trainer.total_timesteps = checkpoint_data.get('timestep', 0)
        trainer.best_elo_rating = checkpoint_data.get('metrics', {}).get('elo_rating', 0)
    
    # start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.logger.logger.info("Training interrupted by user")
        trainer.logger.close()
    except Exception as e:
        trainer.logger.logger.error(f"Training failed: {e}")
        trainer.logger.close()
        raise


if __name__ == "__main__":
    main() 