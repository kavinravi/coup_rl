import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from abc import ABC, abstractmethod
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'envs'))
from coup_game import Action, Role


class BaselineAgent(ABC):
    """abstract base class for baseline agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.reset_state()
    
    @abstractmethod
    def get_action(self, obs: Dict) -> int:
        """get action from observation"""
        pass
    
    def reset_state(self):
        """reset any internal state (called at episode start)"""
        pass
    
    def get_name(self) -> str:
        """get agent name"""
        return self.name


class RandomAgent(BaselineAgent):
    """agent that chooses random valid actions"""
    
    def __init__(self):
        super().__init__("Random")
    
    def get_action(self, obs: Dict) -> int:
        """choose random valid action"""
        valid_actions = np.where(obs["valid_actions"] == 1)[0]
        if len(valid_actions) == 0:
            return 0  # fallback to first action
        return random.choice(valid_actions)


class GreedyAgent(BaselineAgent):
    """agent that prefers high-value actions"""
    
    def __init__(self):
        super().__init__("Greedy")
        
        # action preferences (higher = more preferred)
        self.action_values = {
            Action.INCOME: 1,
            Action.FOREIGN_AID: 2,
            Action.TAX: 5,          # high value but risky
            Action.STEAL: 4,        # good value
            Action.ASSASSINATE: 8,  # high impact
            Action.COUP: 6,         # reliable but expensive
            Action.EXCHANGE: 3,     # utility action
            Action.PASS: 0,         # only when necessary
            Action.CHALLENGE_ACTION: 7,     # depends on situation
            Action.BLOCK_FOREIGN_AID: 2,
            Action.BLOCK_STEAL_CAPTAIN: 3,
            Action.BLOCK_STEAL_AMBASSADOR: 3,
            Action.BLOCK_ASSASSINATE: 9,    # high priority to survive
        }
    
    def get_action(self, obs: Dict) -> int:
        """choose highest value valid action"""
        valid_actions = np.where(obs["valid_actions"] == 1)[0]
        if len(valid_actions) == 0:
            return 0
        
        # map action indices to Action enums
        action_list = list(Action)
        
        # find highest value valid action
        best_action = valid_actions[0]
        best_value = -1
        
        for action_idx in valid_actions:
            if action_idx < len(action_list):
                action = action_list[action_idx]
                value = self.action_values.get(action, 0)
                
                # bonus for actions when we have many coins
                if obs["coins"][obs["current_player"]] >= 7:
                    if action in [Action.COUP, Action.ASSASSINATE]:
                        value += 2
                
                if value > best_value:
                    best_value = value
                    best_action = action_idx
        
        return best_action


class ConservativeAgent(BaselineAgent):
    """agent that plays conservatively - avoids risky actions"""
    
    def __init__(self):
        super().__init__("Conservative")
    
    def get_action(self, obs: Dict) -> int:
        """choose safe actions, avoid risky bluffs"""
        valid_actions = np.where(obs["valid_actions"] == 1)[0]
        if len(valid_actions) == 0:
            return 0
        
        action_list = list(Action)
        my_coins = obs["coins"][obs["current_player"]]
        
        # prefer safe actions
        preferred_order = [
            Action.INCOME,       # always safe
            Action.FOREIGN_AID,  # safe unless blocked
            Action.COUP,         # safe but expensive
            Action.EXCHANGE,     # risky but sometimes necessary
            Action.TAX,          # risky bluff
            Action.STEAL,        # risky bluff
            Action.ASSASSINATE,  # risky and expensive
        ]
        
        # response preferences
        response_order = [
            Action.PASS,                    # usually safest
            Action.BLOCK_ASSASSINATE,       # essential for survival
            Action.BLOCK_FOREIGN_AID,       # low risk
            Action.BLOCK_STEAL_CAPTAIN,     # medium risk
            Action.BLOCK_STEAL_AMBASSADOR,  # medium risk
            Action.CHALLENGE_ACTION,        # risky but sometimes necessary
        ]
        
        # determine if we're responding or acting
        if obs["can_respond"] == 1:
            # we're in response phase
            for action in response_order:
                action_idx = action_list.index(action)
                if action_idx in valid_actions:
                    # special logic for survival
                    if action == Action.BLOCK_ASSASSINATE and obs["pending_action"] != 0:
                        return action_idx  # always try to survive
                    elif action == Action.PASS:
                        return action_idx  # default to passing
        else:
            # we're in action phase
            # if we have enough coins, coup is safe
            if my_coins >= 7:
                coup_idx = action_list.index(Action.COUP)
                if coup_idx in valid_actions:
                    return coup_idx
            
            # otherwise follow preferred order
            for action in preferred_order:
                action_idx = action_list.index(action)
                if action_idx in valid_actions:
                    # only take risky actions if we have low coins
                    if action in [Action.TAX, Action.STEAL] and my_coins >= 4:
                        continue  # skip risky actions when we have options
                    return action_idx
        
        # fallback to first valid action
        return valid_actions[0]


class AggressiveAgent(BaselineAgent):
    """agent that plays aggressively - frequent challenges and bluffs"""
    
    def __init__(self):
        super().__init__("Aggressive")
        self.challenge_rate = 0.4  # 40% chance to challenge
        self.bluff_rate = 0.6      # 60% chance to bluff
    
    def get_action(self, obs: Dict) -> int:
        """choose aggressive actions, challenge and bluff frequently"""
        valid_actions = np.where(obs["valid_actions"] == 1)[0]
        if len(valid_actions) == 0:
            return 0
        
        action_list = list(Action)
        my_coins = obs["coins"][obs["current_player"]]
        
        # in response phase, often challenge
        if obs["can_respond"] == 1:
            challenge_idx = None
            for action_idx in valid_actions:
                if action_idx < len(action_list):
                    action = action_list[action_idx]
                    if action == Action.CHALLENGE_ACTION:
                        challenge_idx = action_idx
                        break
            
            # challenge with some probability
            if challenge_idx is not None and random.random() < self.challenge_rate:
                return challenge_idx
            
            # otherwise pass or block
            for action_idx in valid_actions:
                if action_idx < len(action_list):
                    action = action_list[action_idx]
                    if action == Action.PASS:
                        return action_idx
        
        else:
            # in action phase, prefer aggressive actions
            aggressive_actions = [
                Action.TAX,
                Action.STEAL,
                Action.ASSASSINATE,
                Action.COUP
            ]
            
            # try aggressive actions first
            for action in aggressive_actions:
                action_idx = action_list.index(action)
                if action_idx in valid_actions:
                    # check if we can afford it
                    if action == Action.ASSASSINATE and my_coins < 3:
                        continue
                    if action == Action.COUP and my_coins < 7:
                        continue
                    
                    # bluff check for role actions
                    if action in [Action.TAX, Action.STEAL] and random.random() < self.bluff_rate:
                        return action_idx
                    elif action in [Action.ASSASSINATE, Action.COUP]:
                        return action_idx
            
            # fallback to income/foreign aid
            for action in [Action.FOREIGN_AID, Action.INCOME]:
                action_idx = action_list.index(action)
                if action_idx in valid_actions:
                    return action_idx
        
        return valid_actions[0]


class RuleBasedAgent(BaselineAgent):
    """agent with simple coup strategy rules"""
    
    def __init__(self):
        super().__init__("RuleBased")
        self.opponent_eliminated_cards = {}  # track what opponents have lost
    
    def reset_state(self):
        """reset tracking state"""
        self.opponent_eliminated_cards = {}
    
    def get_action(self, obs: Dict) -> int:
        """choose actions based on simple rules"""
        valid_actions = np.where(obs["valid_actions"] == 1)[0]
        if len(valid_actions) == 0:
            return 0
        
        action_list = list(Action)
        my_coins = obs["coins"][obs["current_player"]]
        current_player = obs["current_player"]
        
        # update tracking info
        self._update_game_state(obs)
        
        if obs["can_respond"] == 1:
            return self._handle_response(obs, valid_actions, action_list)
        else:
            return self._handle_action(obs, valid_actions, action_list, my_coins)
    
    def _update_game_state(self, obs: Dict):
        """update our knowledge of the game state"""
        # track eliminated cards from observation
        for player_id in range(len(obs["coins"])):
            if player_id not in self.opponent_eliminated_cards:
                self.opponent_eliminated_cards[player_id] = []
    
    def _handle_response(self, obs: Dict, valid_actions: List[int], action_list: List[Action]) -> int:
        """handle response phase"""
        # always block assassination attempts
        for action_idx in valid_actions:
            if action_idx < len(action_list):
                action = action_list[action_idx]
                if action == Action.BLOCK_ASSASSINATE:
                    return action_idx
        
        # challenge tax/steal with moderate probability based on game state
        for action_idx in valid_actions:
            if action_idx < len(action_list):
                action = action_list[action_idx]
                if action == Action.CHALLENGE_ACTION:
                    # challenge more often later in game
                    turn_count = obs["turn_count"]
                    challenge_prob = min(0.3 + turn_count * 0.01, 0.6)
                    if random.random() < challenge_prob:
                        return action_idx
        
        # default to pass
        for action_idx in valid_actions:
            if action_idx < len(action_list):
                action = action_list[action_idx]
                if action == Action.PASS:
                    return action_idx
        
        return valid_actions[0]
    
    def _handle_action(self, obs: Dict, valid_actions: List[int], action_list: List[Action], my_coins: int) -> int:
        """handle action phase"""
        # forced coup with 10+ coins
        if my_coins >= 10:
            for action_idx in valid_actions:
                if action_idx < len(action_list):
                    action = action_list[action_idx]
                    if action == Action.COUP:
                        return action_idx
        
        # coup when we have good opportunity (7+ coins and can eliminate someone)
        if my_coins >= 7:
            # check if any opponent has only 1 influence
            for player_id, coins in enumerate(obs["coins"]):
                if player_id != obs["current_player"] and not obs["is_eliminated"][player_id]:
                    # estimate opponent influence (rough heuristic)
                    eliminated_count = np.sum(obs["lost_influence"][player_id])
                    likely_influence = 2 - eliminated_count
                    
                    if likely_influence <= 1:
                        # try to coup this player
                        coup_idx = action_list.index(Action.COUP)
                        if coup_idx in valid_actions:
                            return coup_idx
        
        # assassinate if we have coins and it's worth it
        if my_coins >= 5:  # keep some coins after assassination
            assassinate_idx = action_list.index(Action.ASSASSINATE)
            if assassinate_idx in valid_actions:
                return assassinate_idx
        
        # steal if we need coins
        if my_coins <= 4:
            steal_idx = action_list.index(Action.STEAL)
            if steal_idx in valid_actions:
                return steal_idx
        
        # tax for economic advantage
        if my_coins <= 6:
            tax_idx = action_list.index(Action.TAX)
            if tax_idx in valid_actions:
                return tax_idx
        
        # foreign aid as safer option
        foreign_aid_idx = action_list.index(Action.FOREIGN_AID)
        if foreign_aid_idx in valid_actions:
            return foreign_aid_idx
        
        # fallback to income
        income_idx = action_list.index(Action.INCOME)
        if income_idx in valid_actions:
            return income_idx
        
        return valid_actions[0]


# agent registry for easy access
BASELINE_AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "conservative": ConservativeAgent,
    "aggressive": AggressiveAgent,
    "rule_based": RuleBasedAgent
}


def create_baseline_agent(agent_type: str) -> BaselineAgent:
    """create a baseline agent by name"""
    if agent_type not in BASELINE_AGENTS:
        raise ValueError(f"Unknown baseline agent type: {agent_type}")
    
    return BASELINE_AGENTS[agent_type]()


def get_available_baselines() -> List[str]:
    """get list of available baseline agent types"""
    return list(BASELINE_AGENTS.keys()) 