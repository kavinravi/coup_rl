#!/usr/bin/env python3
"""
Demo script to test the training pipeline with a small example.
This runs a quick training session to verify everything works.
"""

import sys
import os
import yaml
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

from train.train_agent import SelfPlayTrainer


def create_demo_config():
    """Create a minimal configuration for demo"""
    config = {
        'seed': 42,
        'environment': {
            'num_players': 2,
            'max_turns': 50,
            'render_mode': None
        },
        'agent': {
            'device': 'cpu',
            'hidden_size': 64,
            'lstm_size': 32
        },
        'ppo': {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'entropy_coef': 0.01,
            'value_coef': 0.5,
            'max_grad_norm': 0.5,
            'ppo_epochs': 2,
            'batch_size': 32,
            'n_steps': 64,
            'target_kl': 0.01,
            'use_gae': True
        },
        'training': {
            'total_timesteps': 1000,  # Very short for demo
            'eval_frequency': 500,
            'checkpoint_frequency': 1000,
            'log_frequency': 100,
            'early_stopping': {
                'enabled': False,
                'patience': 5,
                'min_delta': 0.01
            },
            'curriculum': {
                'enabled': False,
                'start_players': 2,
                'end_players': 4,
                'progression_threshold': 0.7
            }
        },
        'evaluation': {
            'num_eval_episodes': 10,
            'eval_opponents': ['random', 'greedy'],
            'tournament_size': 20,
            'elo_update': True,
            'detailed_stats': True
        },
        'logging': {
            'tensorboard': False,  # Disable for demo
            'wandb': False,
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'wandb_project': 'coup-rl-demo',
            'wandb_entity': None,
            'wandb_tags': ['demo', 'ppo', 'lstm']
        },
        'checkpoints': {
            'save_dir': 'logs/checkpoints',
            'keep_best': 2,
            'keep_latest': 2,
            'save_optimizer': True,
            'save_training_state': True
        },
        'self_play': {
            'enabled': False,  # Disable for demo
            'opponent_pool_size': 3,
            'update_frequency': 500,
            'selection_strategy': 'recent',
            'league_training': {
                'enabled': False,
                'exploiter_ratio': 0.3,
                'main_agent_ratio': 0.7
            }
        },
        'rewards': {
            'win_reward': 1.0,
            'loss_reward': -1.0,
            'challenge_success': 0.1,
            'challenge_failure': -0.1,
            'successful_bluff': 0.05,
            'failed_bluff': -0.05,
            'gain_coins': 0.01,
            'lose_coins': -0.01,
            'eliminate_opponent': 0.2,
            'lose_influence': -0.15,
            'entropy_bonus': 0.001,
            'turn_penalty': -0.001,
            'action_variety_bonus': 0.001
        },
        'performance': {
            'num_workers': 1,
            'prefetch_batches': 2,
            'pin_memory': True,
            'compile_model': False
        },
        'debug': {
            'enabled': False,
            'log_game_states': False,
            'save_replay_buffer': False,
            'validate_actions': True,
            'profile_training': False
        }
    }
    
    return config


def main():
    """Run demo training"""
    print("=== Coup RL Training Pipeline Demo ===")
    print("This is a quick demo to verify the training pipeline works.")
    print("It will train for only 1000 timesteps with simplified settings.")
    print("")
    
    # Create demo configuration
    config = create_demo_config()
    
    # Create trainer
    print("Creating trainer...")
    trainer = SelfPlayTrainer(config)
    
    print("Starting demo training...")
    print("Expected duration: 1-2 minutes")
    print("")
    
    try:
        # Run training
        trainer.train()
        
        print("")
        print("=== Demo Training Completed Successfully! ===")
        print("Check the logs directory for training results.")
        
        # Show summary
        summary = trainer.logger.get_training_summary()
        print(f"Total timesteps: {summary['total_timesteps']}")
        print(f"Total episodes: {summary['total_episodes']}")
        print(f"Training time: {summary['total_time']:.2f} seconds")
        print(f"Best Elo: {summary['best_performance'].get('elo_rating', 'N/A')}")
        
    except Exception as e:
        print(f"Demo training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 