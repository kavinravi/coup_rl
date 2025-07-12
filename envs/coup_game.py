import random
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import copy


class Role(Enum):
    DUKE = "duke"
    ASSASSIN = "assassin"
    CAPTAIN = "captain"
    AMBASSADOR = "ambassador"
    CONTESSA = "contessa"


class Action(Enum):
    # basic actions
    INCOME = "income"
    FOREIGN_AID = "foreign_aid"
    COUP = "coup"
    
    # role actions
    TAX = "tax"                    # duke
    ASSASSINATE = "assassinate"    # assassin
    STEAL = "steal"               # captain
    EXCHANGE = "exchange"         # ambassador
    
    # reactions
    BLOCK_FOREIGN_AID = "block_foreign_aid"      # duke
    BLOCK_STEAL_CAPTAIN = "block_steal_captain"   # captain
    BLOCK_STEAL_AMBASSADOR = "block_steal_ambassador"  # ambassador
    BLOCK_ASSASSINATE = "block_assassinate"       # contessa
    CHALLENGE_ACTION = "challenge_action"
    CHALLENGE_BLOCK = "challenge_block"
    PASS = "pass"


@dataclass
class ActionData:
    """holds info about an action attempt"""
    action: Action
    actor: int
    target: Optional[int] = None
    claimed_role: Optional[Role] = None
    cost: int = 0
    
    def __post_init__(self):
        # auto-set claimed role and cost based on action
        if self.action == Action.TAX:
            self.claimed_role = Role.DUKE
        elif self.action == Action.ASSASSINATE:
            self.claimed_role = Role.ASSASSIN
            self.cost = 3
        elif self.action == Action.STEAL:
            self.claimed_role = Role.CAPTAIN
        elif self.action == Action.EXCHANGE:
            self.claimed_role = Role.AMBASSADOR
        elif self.action == Action.COUP:
            self.cost = 7
        elif self.action == Action.BLOCK_FOREIGN_AID:
            self.claimed_role = Role.DUKE
        elif self.action == Action.BLOCK_STEAL_CAPTAIN:
            self.claimed_role = Role.CAPTAIN
        elif self.action == Action.BLOCK_STEAL_AMBASSADOR:
            self.claimed_role = Role.AMBASSADOR
        elif self.action == Action.BLOCK_ASSASSINATE:
            self.claimed_role = Role.CONTESSA


@dataclass
class Player:
    """represents a player in the game"""
    id: int
    coins: int = 2
    influence: List[Role] = field(default_factory=list)
    lost_influence: List[Role] = field(default_factory=list)
    is_eliminated: bool = False
    
    def lose_influence(self, role: Role) -> bool:
        """lose an influence card. returns True if player eliminated"""
        if role in self.influence:
            self.influence.remove(role)
            self.lost_influence.append(role)
            
            if len(self.influence) == 0:
                self.is_eliminated = True
                return True
        return False
    
    def has_role(self, role: Role) -> bool:
        """check if player has a specific role"""
        return role in self.influence
    
    def can_afford(self, cost: int) -> bool:
        """check if player can afford an action"""
        return self.coins >= cost


class GameState:
    """main game state class"""
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.players: List[Player] = []
        self.deck: List[Role] = []
        self.current_player = 0
        self.game_over = False
        self.winner: Optional[int] = None
        self.turn_count = 0
        
        # action state tracking
        self.pending_action: Optional[ActionData] = None
        self.waiting_for_responses = False
        self.response_players: Set[int] = set()
        self.responses: List[Tuple[int, Action]] = []
        
        self.initialize_game()
    
    def initialize_game(self):
        """set up the game with deck and players"""
        # create full deck - 3 of each role
        self.deck = [role for role in Role for _ in range(3)]
        random.shuffle(self.deck)
        
        # create players
        for i in range(self.num_players):
            player = Player(id=i)
            # deal 2 cards to each player
            player.influence = [self.deck.pop(), self.deck.pop()]
            self.players.append(player)
    
    def get_active_players(self) -> List[int]:
        """get list of non-eliminated player ids"""
        return [p.id for p in self.players if not p.is_eliminated]
    
    def check_game_over(self) -> bool:
        """check if game is over and set winner"""
        active_players = self.get_active_players()
        if len(active_players) == 1:
            self.game_over = True
            self.winner = active_players[0]
            return True
        return False
    
    def next_player(self):
        """advance to next non-eliminated player"""
        original_player = self.current_player
        while True:
            self.current_player = (self.current_player + 1) % self.num_players
            if not self.players[self.current_player].is_eliminated:
                break
            # prevent infinite loop if somehow all players eliminated
            if self.current_player == original_player:
                break
    
    def can_player_respond(self, player_id: int, action_data: ActionData) -> bool:
        """check if a player can respond to an action (block or challenge)"""
        if player_id == action_data.actor:
            return False  # can't respond to your own action
        
        # check if player can block this action
        if action_data.action == Action.FOREIGN_AID:
            return True  # anyone can challenge, duke can block
        elif action_data.action == Action.STEAL and action_data.target == player_id:
            return True  # target can block with captain/ambassador or challenge
        elif action_data.action == Action.ASSASSINATE and action_data.target == player_id:
            return True  # target can block with contessa or challenge
        elif action_data.action in [Action.TAX, Action.EXCHANGE]:
            return True  # anyone can challenge role actions
        
        return False
    
    def get_response_actions(self, player_id: int, action_data: ActionData) -> List[Action]:
        """get valid response actions for a player"""
        if not self.can_player_respond(player_id, action_data):
            return [Action.PASS]
        
        actions = [Action.PASS, Action.CHALLENGE_ACTION]
        
        # add block options based on action
        if action_data.action == Action.FOREIGN_AID:
            actions.append(Action.BLOCK_FOREIGN_AID)
        elif action_data.action == Action.STEAL and action_data.target == player_id:
            actions.extend([Action.BLOCK_STEAL_CAPTAIN, Action.BLOCK_STEAL_AMBASSADOR])
        elif action_data.action == Action.ASSASSINATE and action_data.target == player_id:
            actions.append(Action.BLOCK_ASSASSINATE)
        
        return actions
    
    def shuffle_role_back(self, role: Role):
        """shuffle a role back into the deck"""
        self.deck.append(role)
        random.shuffle(self.deck)
    
    def draw_card(self) -> Optional[Role]:
        """draw a card from the deck"""
        if self.deck:
            return self.deck.pop()
        return None
    
    def resolve_challenge(self, challenger_id: int, action_data: ActionData) -> bool:
        """resolve a challenge. returns True if challenge succeeds"""
        actor = self.players[action_data.actor]
        challenger = self.players[challenger_id]
        
        # check if actor actually has the claimed role
        if actor.has_role(action_data.claimed_role):
            # challenge failed - challenger loses influence
            if challenger.influence:
                # for now, just remove first influence
                # todo: let player choose which influence to lose
                lost_role = challenger.influence[0]
                challenger.lose_influence(lost_role)
            
            # actor shuffles revealed card back and draws new one
            actor.influence.remove(action_data.claimed_role)
            self.shuffle_role_back(action_data.claimed_role)
            new_card = self.draw_card()
            if new_card:
                actor.influence.append(new_card)
            
            return False  # challenge failed
        else:
            # challenge succeeded - actor loses influence
            if actor.influence:
                lost_role = actor.influence[0]
                actor.lose_influence(lost_role)
            return True  # challenge succeeded
    
    def execute_action(self, action_data: ActionData):
        """execute an action after all challenges/blocks resolved"""
        actor = self.players[action_data.actor]
        
        if action_data.action == Action.INCOME:
            actor.coins += 1
        elif action_data.action == Action.FOREIGN_AID:
            actor.coins += 2
        elif action_data.action == Action.COUP:
            actor.coins -= 7
            if action_data.target is not None:
                target = self.players[action_data.target]
                if target.influence:
                    lost_role = target.influence[0]  # todo: let player choose
                    target.lose_influence(lost_role)
        elif action_data.action == Action.TAX:
            actor.coins += 3
        elif action_data.action == Action.ASSASSINATE:
            actor.coins -= 3
            if action_data.target is not None:
                target = self.players[action_data.target]
                if target.influence:
                    lost_role = target.influence[0]  # todo: let player choose
                    target.lose_influence(lost_role)
        elif action_data.action == Action.STEAL:
            if action_data.target is not None:
                target = self.players[action_data.target]
                stolen = min(2, target.coins)
                target.coins -= stolen
                actor.coins += stolen
        elif action_data.action == Action.EXCHANGE:
            # draw 2 cards, let player choose which to keep
            # for now, just draw 2 and keep all (simplified)
            for _ in range(2):
                new_card = self.draw_card()
                if new_card:
                    actor.influence.append(new_card)
            
            # todo: proper exchange logic where player chooses which cards to keep
    
    def get_valid_actions(self, player_id: int) -> List[Action]:
        """get valid actions for a player"""
        if self.game_over or player_id != self.current_player:
            return []
        
        player = self.players[player_id]
        actions = [Action.INCOME]  # always available
        
        # foreign aid (unless player has 10+ coins)
        if player.coins < 10:
            actions.append(Action.FOREIGN_AID)
        
        # coup (if can afford)
        if player.can_afford(7):
            actions.append(Action.COUP)
        
        # role actions
        actions.extend([Action.TAX, Action.STEAL, Action.EXCHANGE])
        
        # assassinate (if can afford)
        if player.can_afford(3):
            actions.append(Action.ASSASSINATE)
        
        # if player has 10+ coins, must coup
        if player.coins >= 10:
            actions = [Action.COUP]
        
        return actions
    
    def copy(self) -> 'GameState':
        """create a deep copy of the game state"""
        return copy.deepcopy(self)


# some utility functions
def get_action_targets(action: Action, game_state: GameState, actor_id: int) -> List[int]:
    """get valid targets for an action"""
    if action in [Action.COUP, Action.ASSASSINATE, Action.STEAL]:
        # target other players
        return [p.id for p in game_state.players 
                if p.id != actor_id and not p.is_eliminated]
    return []


def action_requires_target(action: Action) -> bool:
    """check if an action requires a target"""
    return action in [Action.COUP, Action.ASSASSINATE, Action.STEAL] 