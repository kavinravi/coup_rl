import time
import random
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'agents'))

from coup_env import CoupEnv
from baseline_agents import BaselineAgent


@dataclass
class MatchResult:
    """result of a single match"""
    player_names: List[str]
    winner: int
    scores: List[float]  # final rewards
    game_length: int
    total_actions: int
    
    # detailed statistics
    player_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    game_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """initialize player stats if not provided"""
        if not self.player_stats:
            for i in range(len(self.player_names)):
                self.player_stats[i] = {
                    'actions_taken': 0,
                    'challenges_made': 0,
                    'challenges_won': 0,
                    'blocks_made': 0,
                    'blocks_successful': 0,
                    'bluffs_attempted': 0,
                    'bluffs_successful': 0,
                    'coins_gained': 0,
                    'coins_spent': 0,
                    'influence_lost': 0
                }


@dataclass 
class TournamentResult:
    """result of a multi-match tournament"""
    player_names: List[str]
    match_results: List[MatchResult]
    win_counts: List[int]
    win_rates: List[float]
    average_scores: List[float]
    
    # aggregate statistics
    total_matches: int
    average_game_length: float
    
    def get_winner(self) -> Tuple[int, str]:
        """get tournament winner"""
        best_player = np.argmax(self.win_rates)
        return best_player, self.player_names[best_player]
    
    def get_head_to_head(self, player1: int, player2: int) -> Dict[str, float]:
        """get head-to-head statistics between two players"""
        h2h_matches = []
        for match in self.match_results:
            if len(match.player_names) == 2:  # only consider 1v1 matches
                if (match.player_names[0] == self.player_names[player1] and 
                    match.player_names[1] == self.player_names[player2]) or \
                   (match.player_names[0] == self.player_names[player2] and 
                    match.player_names[1] == self.player_names[player1]):
                    h2h_matches.append(match)
        
        if not h2h_matches:
            return {'matches': 0, 'wins': 0, 'win_rate': 0.0}
        
        player1_wins = 0
        for match in h2h_matches:
            if match.player_names[match.winner] == self.player_names[player1]:
                player1_wins += 1
        
        return {
            'matches': len(h2h_matches),
            'wins': player1_wins,
            'win_rate': player1_wins / len(h2h_matches)
        }


class MatchRunner:
    """runs matches between different types of agents"""
    
    def __init__(self, num_players: int = 2, max_steps: int = 200, verbose: bool = False):
        self.num_players = num_players
        self.max_steps = max_steps
        self.verbose = verbose
        
        # create environment
        self.env = CoupEnv(num_players=num_players, max_turns=max_steps)
        
    def run_match(self, agents: List[Any], seed: Optional[int] = None) -> MatchResult:
        """run a single match between agents"""
        if len(agents) != self.num_players:
            raise ValueError(f"Expected {self.num_players} agents, got {len(agents)}")
        
        # set seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # reset environment and agents
        obs, info = self.env.reset(seed=seed)
        
        for agent in agents:
            if hasattr(agent, 'reset_state'):
                agent.reset_state()
            if hasattr(agent, 'reset_hidden_state'):
                agent.reset_hidden_state()
        
        # game state tracking
        episode_rewards = [0.0] * self.num_players
        game_history = []
        step_count = 0
        
        # player statistics
        player_stats = {}
        for i in range(self.num_players):
            player_stats[i] = {
                'actions_taken': 0,
                'challenges_made': 0,
                'challenges_won': 0,
                'blocks_made': 0,
                'blocks_successful': 0,
                'bluffs_attempted': 0,
                'bluffs_successful': 0,
                'coins_gained': 0,
                'coins_spent': 0,
                'influence_lost': 0
            }
        
        if self.verbose:
            print(f"Starting match: {[self._get_agent_name(agent) for agent in agents]}")
        
        # main game loop
        while step_count < self.max_steps:
            current_player = info.get('current_player', 0)
            
            # get action from current agent
            try:
                action = self._get_agent_action(agents[current_player], obs)
            except Exception as e:
                if self.verbose:
                    print(f"Error getting action from agent {current_player}: {e}")
                action = 0  # fallback to first valid action
            
            # store action in history
            game_history.append({
                'step': step_count,
                'player': current_player,
                'action': action,
                'obs': obs.copy() if isinstance(obs, dict) else obs
            })
            
            # step environment
            prev_obs = obs.copy() if isinstance(obs, dict) else obs
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # update statistics
            self._update_player_stats(player_stats[current_player], action, prev_obs, obs, reward)
            
            # track rewards
            episode_rewards[current_player] += reward
            step_count += 1
            
            if self.verbose and step_count % 10 == 0:
                print(f"Step {step_count}, Current player: {current_player}, Reward: {reward:.3f}")
            
            # check for game end
            if terminated or truncated:
                winner = info.get('winner', None)
                if winner is None:
                    # fallback to highest scoring player
                    winner = int(np.argmax(episode_rewards))
                if self.verbose:
                    print(f"Game ended after {step_count} steps. Winner: {winner} ({self._get_agent_name(agents[winner])})")
                break
        else:
            # game reached max steps
            winner = int(np.argmax(episode_rewards))
            if self.verbose:
                print(f"Game reached max steps ({self.max_steps}). Winner: {winner} by score")
        
        # create match result
        agent_names = [self._get_agent_name(agent) for agent in agents]
        
        return MatchResult(
            player_names=agent_names,
            winner=winner,
            scores=episode_rewards,
            game_length=step_count,
            total_actions=step_count,
            player_stats=player_stats,
            game_history=game_history
        )
    
    def run_tournament(self, agents: List[Any], num_matches: int = 100, 
                      progress_callback: Optional[callable] = None) -> TournamentResult:
        """run a tournament with multiple matches"""
        if len(agents) != self.num_players:
            raise ValueError(f"Expected {self.num_players} agents, got {len(agents)}")
        
        agent_names = [self._get_agent_name(agent) for agent in agents]
        match_results = []
        win_counts = [0] * self.num_players
        total_scores = [0.0] * self.num_players
        
        if self.verbose:
            print(f"Starting tournament: {num_matches} matches between {agent_names}")
        
        for match_idx in range(num_matches):
            # run match with different seed each time
            match_result = self.run_match(agents, seed=match_idx)
            match_results.append(match_result)
            
            # update statistics
            win_counts[match_result.winner] += 1
            for i, score in enumerate(match_result.scores):
                total_scores[i] += score
            
            # progress callback
            if progress_callback:
                progress_callback(match_idx + 1, num_matches, match_result)
            
            if self.verbose and (match_idx + 1) % 10 == 0:
                current_win_rates = [w / (match_idx + 1) for w in win_counts]
                print(f"Match {match_idx + 1}/{num_matches}: Win rates: {current_win_rates}")
        
        # compute final statistics
        win_rates = [w / num_matches for w in win_counts]
        average_scores = [s / num_matches for s in total_scores]
        average_game_length = np.mean([m.game_length for m in match_results])
        
        return TournamentResult(
            player_names=agent_names,
            match_results=match_results,
            win_counts=win_counts,
            win_rates=win_rates,
            average_scores=average_scores,
            total_matches=num_matches,
            average_game_length=average_game_length
        )
    
    def run_round_robin(self, agents: List[Any], matches_per_pair: int = 50) -> Dict[str, TournamentResult]:
        """run round robin tournament between all agents"""
        if self.num_players != 2:
            raise ValueError("Round robin only supported for 2-player games")
        
        results = {}
        agent_names = [self._get_agent_name(agent) for agent in agents]
        
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                pair_name = f"{agent_names[i]}_vs_{agent_names[j]}"
                
                if self.verbose:
                    print(f"Running {pair_name}: {matches_per_pair} matches")
                
                tournament_result = self.run_tournament([agent1, agent2], matches_per_pair)
                results[pair_name] = tournament_result
        
        return results
    
    def _get_agent_name(self, agent: Any) -> str:
        """get agent name for display"""
        if hasattr(agent, 'get_name'):
            return agent.get_name()
        elif hasattr(agent, 'name'):
            return agent.name
        elif hasattr(agent, '__class__'):
            return agent.__class__.__name__
        else:
            return str(type(agent))
    
    def _get_agent_action(self, agent: Any, obs: Dict) -> int:
        """get action from agent (handles different agent types)"""
        if hasattr(agent, 'get_action'):
            if isinstance(agent, BaselineAgent):
                return agent.get_action(obs)
            else:
                # assume it's our PPO agent
                action, _ = agent.get_action(obs)
                return action
        else:
            raise ValueError(f"Agent {agent} does not have get_action method")
    
    def _update_player_stats(self, stats: Dict, action: int, prev_obs: Dict, obs: Dict, reward: float):
        """update player statistics based on action and outcome"""
        stats['actions_taken'] += 1
        
        # track coin changes
        prev_coins = prev_obs.get('coins', [0, 0])[prev_obs.get('current_player', 0)]
        curr_coins = obs.get('coins', [0, 0])[prev_obs.get('current_player', 0)]
        
        if curr_coins > prev_coins:
            stats['coins_gained'] += (curr_coins - prev_coins)
        elif curr_coins < prev_coins:
            stats['coins_spent'] += (prev_coins - curr_coins)
        
        # track influence changes (simplified)
        if reward < -0.5:  # approximate detection of influence loss
            stats['influence_lost'] += 1
        
        # todo: add more detailed action analysis based on action type
        # this would require tracking the action history and outcomes


# utility functions
def quick_evaluation(agent1: Any, agent2: Any, num_matches: int = 100, verbose: bool = True) -> TournamentResult:
    """quick head-to-head evaluation between two agents"""
    runner = MatchRunner(num_players=2, verbose=verbose)
    return runner.run_tournament([agent1, agent2], num_matches)


def baseline_benchmark(agent: Any, num_matches: int = 50, verbose: bool = True) -> Dict[str, float]:
    """benchmark an agent against all baseline agents"""
    from baseline_agents import get_available_baselines, create_baseline_agent
    
    results = {}
    runner = MatchRunner(num_players=2, verbose=verbose)
    
    for baseline_name in get_available_baselines():
        baseline_agent = create_baseline_agent(baseline_name)
        tournament = runner.run_tournament([agent, baseline_agent], num_matches)
        results[baseline_name] = tournament.win_rates[0]  # win rate of our agent
        
        if verbose:
            print(f"vs {baseline_name}: {tournament.win_rates[0]:.3f} win rate")
    
    return results 