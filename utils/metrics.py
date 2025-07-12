import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
from coup_game import Action, Role


class GameStatistics:
    """collect and compute game statistics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """reset all statistics"""
        self.episodes = []
        self.current_episode = None
        
    def start_episode(self, num_players: int):
        """start tracking a new episode"""
        self.current_episode = {
            'num_players': num_players,
            'length': 0,
            'winner': None,
            'players': {i: self._init_player_stats() for i in range(num_players)},
            'actions': [],
            'challenges': [],
            'blocks': [],
            'eliminations': []
        }
    
    def _init_player_stats(self) -> Dict[str, Any]:
        """initialize player statistics"""
        return {
            'actions_taken': 0,
            'challenges_made': 0,
            'challenges_won': 0,
            'blocks_made': 0,
            'blocks_successful': 0,
            'bluffs_attempted': 0,
            'bluffs_successful': 0,
            'coins_gained': 0,
            'coins_spent': 0,
            'influence_lost': 0,
            'final_coins': 0,
            'final_influence': 0,
            'actions_distribution': defaultdict(int),
            'eliminated_turn': None
        }
    
    def record_action(self, player: int, action: Action, target: Optional[int] = None,
                     success: bool = True, was_bluff: bool = False):
        """record an action taken by a player"""
        if self.current_episode is None:
            return
        
        self.current_episode['actions'].append({
            'player': player,
            'action': action,
            'target': target,
            'success': success,
            'was_bluff': was_bluff,
            'turn': self.current_episode['length']
        })
        
        # update player stats
        player_stats = self.current_episode['players'][player]
        player_stats['actions_taken'] += 1
        player_stats['actions_distribution'][action] += 1
        
        if was_bluff:
            player_stats['bluffs_attempted'] += 1
            if success:
                player_stats['bluffs_successful'] += 1
    
    def record_challenge(self, challenger: int, challenged: int, action: Action,
                        challenger_won: bool, had_card: bool):
        """record a challenge"""
        if self.current_episode is None:
            return
        
        self.current_episode['challenges'].append({
            'challenger': challenger,
            'challenged': challenged,
            'action': action,
            'challenger_won': challenger_won,
            'had_card': had_card,
            'turn': self.current_episode['length']
        })
        
        # update player stats
        challenger_stats = self.current_episode['players'][challenger]
        challenger_stats['challenges_made'] += 1
        if challenger_won:
            challenger_stats['challenges_won'] += 1
    
    def record_block(self, blocker: int, blocked_player: int, action: Action,
                    success: bool, was_bluff: bool):
        """record a block"""
        if self.current_episode is None:
            return
        
        self.current_episode['blocks'].append({
            'blocker': blocker,
            'blocked_player': blocked_player,
            'action': action,
            'success': success,
            'was_bluff': was_bluff,
            'turn': self.current_episode['length']
        })
        
        # update player stats
        blocker_stats = self.current_episode['players'][blocker]
        blocker_stats['blocks_made'] += 1
        if success:
            blocker_stats['blocks_successful'] += 1
        
        if was_bluff:
            blocker_stats['bluffs_attempted'] += 1
            if success:
                blocker_stats['bluffs_successful'] += 1
    
    def record_elimination(self, player: int, eliminated_by: Optional[int] = None):
        """record player elimination"""
        if self.current_episode is None:
            return
        
        self.current_episode['eliminations'].append({
            'player': player,
            'eliminated_by': eliminated_by,
            'turn': self.current_episode['length']
        })
        
        # update player stats
        player_stats = self.current_episode['players'][player]
        player_stats['eliminated_turn'] = self.current_episode['length']
    
    def record_coins_change(self, player: int, change: int):
        """record coins gained/lost"""
        if self.current_episode is None:
            return
        
        player_stats = self.current_episode['players'][player]
        if change > 0:
            player_stats['coins_gained'] += change
        else:
            player_stats['coins_spent'] += abs(change)
    
    def record_influence_lost(self, player: int):
        """record influence lost"""
        if self.current_episode is None:
            return
        
        player_stats = self.current_episode['players'][player]
        player_stats['influence_lost'] += 1
    
    def finish_episode(self, winner: int, final_states: Dict[int, Dict[str, Any]]):
        """finish current episode"""
        if self.current_episode is None:
            return
        
        self.current_episode['winner'] = winner
        self.current_episode['length'] = len(self.current_episode['actions'])
        
        # update final states
        for player_id, state in final_states.items():
            if player_id in self.current_episode['players']:
                self.current_episode['players'][player_id]['final_coins'] = state.get('coins', 0)
                self.current_episode['players'][player_id]['final_influence'] = state.get('influence', 0)
        
        # add to episodes
        self.episodes.append(self.current_episode)
        self.current_episode = None
    
    def get_aggregate_stats(self) -> Dict[str, Any]:
        """get aggregated statistics across all episodes"""
        if not self.episodes:
            return {}
        
        stats = {
            'total_episodes': len(self.episodes),
            'avg_episode_length': np.mean([ep['length'] for ep in self.episodes]),
            'episode_length_std': np.std([ep['length'] for ep in self.episodes]),
            'min_episode_length': np.min([ep['length'] for ep in self.episodes]),
            'max_episode_length': np.max([ep['length'] for ep in self.episodes]),
        }
        
        # collect all player stats
        all_player_stats = []
        for episode in self.episodes:
            for player_stats in episode['players'].values():
                all_player_stats.append(player_stats)
        
        if all_player_stats:
            # action statistics
            stats['avg_actions_per_player'] = np.mean([p['actions_taken'] for p in all_player_stats])
            stats['avg_challenges_per_player'] = np.mean([p['challenges_made'] for p in all_player_stats])
            stats['avg_blocks_per_player'] = np.mean([p['blocks_made'] for p in all_player_stats])
            
            # success rates
            challenge_rates = [p['challenges_won'] / max(1, p['challenges_made']) for p in all_player_stats]
            stats['avg_challenge_success_rate'] = np.mean(challenge_rates)
            
            block_rates = [p['blocks_successful'] / max(1, p['blocks_made']) for p in all_player_stats]
            stats['avg_block_success_rate'] = np.mean(block_rates)
            
            bluff_rates = [p['bluffs_successful'] / max(1, p['bluffs_attempted']) for p in all_player_stats]
            stats['avg_bluff_success_rate'] = np.mean(bluff_rates)
            
            # resource statistics
            stats['avg_coins_gained'] = np.mean([p['coins_gained'] for p in all_player_stats])
            stats['avg_coins_spent'] = np.mean([p['coins_spent'] for p in all_player_stats])
            stats['avg_influence_lost'] = np.mean([p['influence_lost'] for p in all_player_stats])
            
            # action distribution
            all_actions = defaultdict(int)
            total_actions = 0
            for player_stats in all_player_stats:
                for action, count in player_stats['actions_distribution'].items():
                    all_actions[action] += count
                    total_actions += count
            
            if total_actions > 0:
                stats['action_distribution'] = {
                    action.name: count / total_actions for action, count in all_actions.items()
                }
            
        return stats
    
    def get_player_performance(self, player_id: int) -> Dict[str, Any]:
        """get performance statistics for a specific player"""
        player_episodes = []
        wins = 0
        
        for episode in self.episodes:
            if player_id in episode['players']:
                player_episodes.append(episode['players'][player_id])
                if episode['winner'] == player_id:
                    wins += 1
        
        if not player_episodes:
            return {}
        
        stats = {
            'total_games': len(player_episodes),
            'wins': wins,
            'win_rate': wins / len(player_episodes),
            'avg_actions_per_game': np.mean([p['actions_taken'] for p in player_episodes]),
            'avg_challenges_per_game': np.mean([p['challenges_made'] for p in player_episodes]),
            'avg_blocks_per_game': np.mean([p['blocks_made'] for p in player_episodes]),
            'challenge_success_rate': np.mean([p['challenges_won'] / max(1, p['challenges_made']) for p in player_episodes]),
            'block_success_rate': np.mean([p['blocks_successful'] / max(1, p['blocks_made']) for p in player_episodes]),
            'bluff_success_rate': np.mean([p['bluffs_successful'] / max(1, p['bluffs_attempted']) for p in player_episodes]),
            'avg_coins_gained': np.mean([p['coins_gained'] for p in player_episodes]),
            'avg_coins_spent': np.mean([p['coins_spent'] for p in player_episodes]),
            'avg_influence_lost': np.mean([p['influence_lost'] for p in player_episodes]),
            'avg_final_coins': np.mean([p['final_coins'] for p in player_episodes]),
            'avg_final_influence': np.mean([p['final_influence'] for p in player_episodes]),
        }
        
        return stats


class PerformanceMetrics:
    """compute performance metrics for evaluation"""
    
    @staticmethod
    def compute_challenge_accuracy(challenges: List[Dict[str, Any]]) -> float:
        """compute accuracy of challenges"""
        if not challenges:
            return 0.0
        
        correct_challenges = sum(1 for c in challenges if c['challenger_won'] == (not c['had_card']))
        return correct_challenges / len(challenges)
    
    @staticmethod
    def compute_block_accuracy(blocks: List[Dict[str, Any]]) -> float:
        """compute accuracy of blocks"""
        if not blocks:
            return 0.0
        
        successful_blocks = sum(1 for b in blocks if b['success'])
        return successful_blocks / len(blocks)
    
    @staticmethod
    def compute_bluff_success_rate(actions: List[Dict[str, Any]]) -> float:
        """compute success rate of bluffs"""
        bluffs = [a for a in actions if a['was_bluff']]
        if not bluffs:
            return 0.0
        
        successful_bluffs = sum(1 for b in bluffs if b['success'])
        return successful_bluffs / len(bluffs)
    
    @staticmethod
    def compute_action_distribution(actions: List[Dict[str, Any]]) -> Dict[str, float]:
        """compute distribution of actions taken"""
        if not actions:
            return {}
        
        action_counts = Counter(a['action'] for a in actions)
        total_actions = len(actions)
        
        return {action.name: count / total_actions for action, count in action_counts.items()}
    
    @staticmethod
    def compute_elimination_efficiency(eliminations: List[Dict[str, Any]], 
                                     total_turns: int) -> float:
        """compute how efficiently players eliminate opponents"""
        if not eliminations or total_turns == 0:
            return 0.0
        
        return len(eliminations) / total_turns
    
    @staticmethod
    def compute_resource_efficiency(player_stats: Dict[str, Any]) -> float:
        """compute how efficiently a player uses resources"""
        coins_spent = player_stats.get('coins_spent', 0)
        eliminations = player_stats.get('eliminations_caused', 0)
        
        if coins_spent == 0:
            return 0.0
        
        return eliminations / coins_spent
    
    @staticmethod
    def compute_survival_rate(players: List[Dict[str, Any]]) -> float:
        """compute survival rate (not eliminated)"""
        if not players:
            return 0.0
        
        survived = sum(1 for p in players if p.get('eliminated_turn') is None)
        return survived / len(players)
    
    @staticmethod
    def compute_aggressive_play_score(player_stats: Dict[str, Any]) -> float:
        """compute how aggressively a player plays"""
        actions_taken = player_stats.get('actions_taken', 0)
        challenges_made = player_stats.get('challenges_made', 0)
        bluffs_attempted = player_stats.get('bluffs_attempted', 0)
        
        if actions_taken == 0:
            return 0.0
        
        aggressive_actions = challenges_made + bluffs_attempted
        return aggressive_actions / actions_taken
    
    @staticmethod
    def compute_conservative_play_score(player_stats: Dict[str, Any]) -> float:
        """compute how conservatively a player plays"""
        actions_taken = player_stats.get('actions_taken', 0)
        safe_actions = player_stats.get('actions_distribution', {}).get(Action.INCOME, 0)
        safe_actions += player_stats.get('actions_distribution', {}).get(Action.FOREIGN_AID, 0)
        
        if actions_taken == 0:
            return 0.0
        
        return safe_actions / actions_taken


class EvaluationMetrics:
    """compute comprehensive evaluation metrics"""
    
    def __init__(self):
        self.game_stats = GameStatistics()
        self.performance_metrics = PerformanceMetrics()
    
    def evaluate_games(self, game_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """evaluate a batch of game results"""
        metrics = {}
        
        # basic game statistics
        if game_results:
            episode_lengths = [r['length'] for r in game_results]
            metrics['avg_episode_length'] = np.mean(episode_lengths)
            metrics['episode_length_std'] = np.std(episode_lengths)
            metrics['min_episode_length'] = np.min(episode_lengths)
            metrics['max_episode_length'] = np.max(episode_lengths)
        
        # collect all actions, challenges, blocks
        all_actions = []
        all_challenges = []
        all_blocks = []
        
        for result in game_results:
            all_actions.extend(result.get('actions', []))
            all_challenges.extend(result.get('challenges', []))
            all_blocks.extend(result.get('blocks', []))
        
        # compute performance metrics
        metrics['challenge_accuracy'] = self.performance_metrics.compute_challenge_accuracy(all_challenges)
        metrics['block_accuracy'] = self.performance_metrics.compute_block_accuracy(all_blocks)
        metrics['bluff_success_rate'] = self.performance_metrics.compute_bluff_success_rate(all_actions)
        metrics['action_distribution'] = self.performance_metrics.compute_action_distribution(all_actions)
        
        # compute player-specific metrics
        player_metrics = defaultdict(list)
        for result in game_results:
            for player_id, player_stats in result.get('players', {}).items():
                player_metrics[player_id].append(player_stats)
        
        # aggregate player metrics
        for player_id, stats_list in player_metrics.items():
            if stats_list:
                metrics[f'player_{player_id}_avg_coins'] = np.mean([s['final_coins'] for s in stats_list])
                metrics[f'player_{player_id}_avg_influence'] = np.mean([s['final_influence'] for s in stats_list])
                metrics[f'player_{player_id}_aggressive_score'] = np.mean([
                    self.performance_metrics.compute_aggressive_play_score(s) for s in stats_list
                ])
                metrics[f'player_{player_id}_conservative_score'] = np.mean([
                    self.performance_metrics.compute_conservative_play_score(s) for s in stats_list
                ])
        
        return metrics
    
    def compare_agents(self, agent_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """compare performance between different agents"""
        comparison = {}
        
        for agent_name, results in agent_results.items():
            agent_metrics = self.evaluate_games(results)
            comparison[agent_name] = agent_metrics
        
        # compute relative performance
        if len(agent_results) > 1:
            agent_names = list(agent_results.keys())
            
            # win rate matrix
            win_matrix = {}
            for agent1 in agent_names:
                win_matrix[agent1] = {}
                for agent2 in agent_names:
                    if agent1 != agent2:
                        # compute head-to-head win rate
                        wins = 0
                        total = 0
                        for result in agent_results[agent1]:
                            if result.get('opponent') == agent2:
                                if result.get('winner') == agent1:
                                    wins += 1
                                total += 1
                        
                        win_matrix[agent1][agent2] = wins / total if total > 0 else 0.0
            
            comparison['win_matrix'] = win_matrix
        
        return comparison 