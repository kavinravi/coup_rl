import math
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json
import os


@dataclass
class EloMatch:
    """record of a single elo-rated match"""
    timestamp: float
    player1: str
    player2: str
    winner: str  # name of winning player
    player1_rating_before: float
    player2_rating_before: float
    player1_rating_after: float
    player2_rating_after: float
    rating_change: float  # positive for player1, negative for player2
    k_factor: float
    
    def to_dict(self) -> Dict:
        """convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'player1': self.player1,
            'player2': self.player2,
            'winner': self.winner,
            'player1_rating_before': self.player1_rating_before,
            'player2_rating_before': self.player2_rating_before,
            'player1_rating_after': self.player1_rating_after,
            'player2_rating_after': self.player2_rating_after,
            'rating_change': self.rating_change,
            'k_factor': self.k_factor
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EloMatch':
        """create from dictionary"""
        return cls(**data)


@dataclass
class PlayerStats:
    """statistics for a single player"""
    name: str
    current_rating: float = 1500.0
    peak_rating: float = 1500.0
    matches_played: int = 0
    wins: int = 0
    losses: int = 0
    win_streak: int = 0
    loss_streak: int = 0
    rating_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, rating)
    
    def update_rating(self, new_rating: float, won: bool):
        """update player rating and statistics"""
        self.current_rating = new_rating
        self.peak_rating = max(self.peak_rating, new_rating)
        self.matches_played += 1
        
        if won:
            self.wins += 1
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.losses += 1
            self.loss_streak += 1
            self.win_streak = 0
        
        # record rating history
        self.rating_history.append((time.time(), new_rating))
    
    @property
    def win_rate(self) -> float:
        """calculate win rate"""
        if self.matches_played == 0:
            return 0.0
        return self.wins / self.matches_played
    
    def to_dict(self) -> Dict:
        """convert to dictionary for serialization"""
        return {
            'name': self.name,
            'current_rating': self.current_rating,
            'peak_rating': self.peak_rating,
            'matches_played': self.matches_played,
            'wins': self.wins,
            'losses': self.losses,
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'rating_history': self.rating_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PlayerStats':
        """create from dictionary"""
        return cls(**data)


class EloRatingSystem:
    """elo rating system for tracking agent performance"""
    
    def __init__(self, initial_rating: float = 1500.0, k_factor: float = 32.0):
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        
        # player tracking
        self.players: Dict[str, PlayerStats] = {}
        self.match_history: List[EloMatch] = []
        
        # rating thresholds for categorization
        self.rating_categories = {
            'Beginner': (0, 1200),
            'Novice': (1200, 1400),
            'Intermediate': (1400, 1600),
            'Advanced': (1600, 1800),
            'Expert': (1800, 2000),
            'Master': (2000, float('inf'))
        }
    
    def get_player(self, name: str) -> PlayerStats:
        """get or create player stats"""
        if name not in self.players:
            self.players[name] = PlayerStats(name=name, current_rating=self.initial_rating)
        return self.players[name]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """calculate expected score for player a against player b"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, player1_name: str, player2_name: str, winner_name: str, k_factor: Optional[float] = None) -> EloMatch:
        """update ratings after a match"""
        if k_factor is None:
            k_factor = self.k_factor
        
        # get or create players
        player1 = self.get_player(player1_name)
        player2 = self.get_player(player2_name)
        
        # store ratings before update
        rating1_before = player1.current_rating
        rating2_before = player2.current_rating
        
        # calculate expected scores
        expected1 = self.expected_score(rating1_before, rating2_before)
        expected2 = 1 - expected1
        
        # actual scores (1 for win, 0 for loss)
        if winner_name == player1_name:
            score1, score2 = 1.0, 0.0
        elif winner_name == player2_name:
            score1, score2 = 0.0, 1.0
        else:
            # draw (if supported)
            score1, score2 = 0.5, 0.5
        
        # calculate rating changes
        rating_change1 = k_factor * (score1 - expected1)
        rating_change2 = k_factor * (score2 - expected2)
        
        # update ratings
        new_rating1 = rating1_before + rating_change1
        new_rating2 = rating2_before + rating_change2
        
        # update player stats
        player1.update_rating(new_rating1, score1 > score2)
        player2.update_rating(new_rating2, score2 > score1)
        
        # create match record
        match_record = EloMatch(
            timestamp=time.time(),
            player1=player1_name,
            player2=player2_name,
            winner=winner_name,
            player1_rating_before=rating1_before,
            player2_rating_before=rating2_before,
            player1_rating_after=new_rating1,
            player2_rating_after=new_rating2,
            rating_change=rating_change1,
            k_factor=k_factor
        )
        
        self.match_history.append(match_record)
        return match_record
    
    def get_leaderboard(self, limit: Optional[int] = None) -> List[PlayerStats]:
        """get leaderboard sorted by rating"""
        sorted_players = sorted(self.players.values(), key=lambda p: p.current_rating, reverse=True)
        if limit:
            return sorted_players[:limit]
        return sorted_players
    
    def get_player_category(self, player_name: str) -> str:
        """get rating category for a player"""
        player = self.get_player(player_name)
        rating = player.current_rating
        
        for category, (min_rating, max_rating) in self.rating_categories.items():
            if min_rating <= rating < max_rating:
                return category
        
        return "Unknown"
    
    def get_head_to_head(self, player1: str, player2: str) -> Dict:
        """get head-to-head statistics between two players"""
        matches = [m for m in self.match_history 
                  if (m.player1 == player1 and m.player2 == player2) or 
                     (m.player1 == player2 and m.player2 == player1)]
        
        if not matches:
            return {'matches': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0}
        
        wins = sum(1 for m in matches if m.winner == player1)
        losses = len(matches) - wins
        
        return {
            'matches': len(matches),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(matches) if matches else 0.0
        }
    
    def simulate_match_outcome(self, player1: str, player2: str) -> Dict[str, float]:
        """predict match outcome probabilities"""
        p1 = self.get_player(player1)
        p2 = self.get_player(player2)
        
        p1_win_prob = self.expected_score(p1.current_rating, p2.current_rating)
        p2_win_prob = 1 - p1_win_prob
        
        return {
            player1: p1_win_prob,
            player2: p2_win_prob
        }
    
    def get_rating_trend(self, player_name: str, recent_matches: int = 10) -> str:
        """get recent rating trend for a player"""
        player = self.get_player(player_name)
        
        if len(player.rating_history) < 2:
            return "Stable"
        
        recent_history = player.rating_history[-recent_matches:]
        if len(recent_history) < 2:
            return "Stable"
        
        rating_change = recent_history[-1][1] - recent_history[0][1]
        
        if rating_change > 50:
            return "Rising"
        elif rating_change < -50:
            return "Falling"
        else:
            return "Stable"
    
    def export_data(self, filepath: str):
        """export all data to json file"""
        data = {
            'players': {name: player.to_dict() for name, player in self.players.items()},
            'match_history': [match.to_dict() for match in self.match_history],
            'config': {
                'initial_rating': self.initial_rating,
                'k_factor': self.k_factor,
                'rating_categories': self.rating_categories
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def import_data(self, filepath: str):
        """import data from json file"""
        if not os.path.exists(filepath):
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # import players
        self.players = {}
        for name, player_data in data.get('players', {}).items():
            self.players[name] = PlayerStats.from_dict(player_data)
        
        # import match history
        self.match_history = []
        for match_data in data.get('match_history', []):
            self.match_history.append(EloMatch.from_dict(match_data))
        
        # import config
        config = data.get('config', {})
        self.initial_rating = config.get('initial_rating', self.initial_rating)
        self.k_factor = config.get('k_factor', self.k_factor)
        self.rating_categories = config.get('rating_categories', self.rating_categories)
    
    def print_leaderboard(self, limit: int = 10):
        """print formatted leaderboard"""
        leaderboard = self.get_leaderboard(limit)
        
        print(f"\n{'='*60}")
        print(f"{'ELO LEADERBOARD':^60}")
        print(f"{'='*60}")
        print(f"{'Rank':<5} {'Player':<15} {'Rating':<8} {'Category':<12} {'W-L':<8} {'Win%':<6} {'Trend':<8}")
        print("-" * 60)
        
        for i, player in enumerate(leaderboard, 1):
            category = self.get_player_category(player.name)
            trend = self.get_rating_trend(player.name)
            win_loss = f"{player.wins}-{player.losses}"
            win_pct = f"{player.win_rate:.1%}"
            
            print(f"{i:<5} {player.name:<15} {player.current_rating:<8.0f} {category:<12} {win_loss:<8} {win_pct:<6} {trend:<8}")
        
        print(f"{'='*60}")


# utility functions for integration with match runner
def update_elo_from_tournament(elo_system: EloRatingSystem, tournament_result, k_factor: Optional[float] = None):
    """update elo ratings from tournament results"""
    for match in tournament_result.match_results:
        if len(match.player_names) == 2:  # only process 1v1 matches
            winner_name = match.player_names[match.winner]
            player1, player2 = match.player_names[0], match.player_names[1]
            elo_system.update_ratings(player1, player2, winner_name, k_factor)


def run_elo_tournament(agents: List, elo_system: EloRatingSystem, num_matches: int = 100, verbose: bool = True):
    """run tournament and update elo ratings"""
    from match_runner import MatchRunner
    
    runner = MatchRunner(num_players=2, verbose=False)
    
    if verbose:
        print(f"Running Elo tournament: {num_matches} matches")
    
    # run round robin between all agent pairs
    results = runner.run_round_robin(agents, matches_per_pair=num_matches // (len(agents) * (len(agents) - 1) // 2))
    
    # update elo ratings
    for pair_name, tournament_result in results.items():
        update_elo_from_tournament(elo_system, tournament_result)
    
    if verbose:
        elo_system.print_leaderboard()
    
    return elo_system 