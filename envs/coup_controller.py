from typing import List, Optional, Dict, Tuple, Set
from enum import Enum
from coup_game import GameState, Action, Role, ActionData, Player
import random


class GamePhase(Enum):
    ACTION = "action"           # waiting for primary action
    RESPONSE = "response"       # waiting for responses (blocks/challenges)
    CHALLENGE = "challenge"     # resolving challenges
    BLOCK = "block"            # resolving blocks
    EXECUTION = "execution"     # executing the action
    CHOICE = "choice"          # waiting for player choice (which card to lose, etc.)


class ResponseType(Enum):
    CHALLENGE_ACTION = "challenge_action"
    CHALLENGE_BLOCK = "challenge_block"
    BLOCK = "block"
    PASS = "pass"


class CoupController:
    """manages the full game flow including actions, responses, challenges, and blocks"""
    
    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        self.game_state = GameState(num_players)
        self.phase = GamePhase.ACTION
        
        # action flow tracking
        self.pending_action: Optional[ActionData] = None
        self.pending_responses: List[Tuple[int, ResponseType, Optional[Role]]] = []
        self.response_deadline: Set[int] = set()  # players who can still respond
        
        # choice tracking
        self.waiting_for_choice: Optional[int] = None
        self.choice_type: Optional[str] = None
        self.choice_options: List = []
        
        # history for learning
        self.action_history: List[Dict] = []
        self.turn_history: List[Dict] = []
    
    def get_current_phase(self) -> GamePhase:
        """get the current game phase"""
        return self.phase
    
    def get_current_player(self) -> int:
        """get the current player whose turn it is"""
        return self.game_state.current_player
    
    def get_active_players(self) -> List[int]:
        """get list of active (non-eliminated) players"""
        return self.game_state.get_active_players()
    
    def is_game_over(self) -> bool:
        """check if the game is over"""
        return self.game_state.game_over
    
    def get_winner(self) -> Optional[int]:
        """get the winner if game is over"""
        return self.game_state.winner
    
    def get_valid_actions(self, player_id: int) -> List[Action]:
        """get valid actions for a player based on current phase"""
        if self.phase == GamePhase.ACTION:
            if player_id == self.game_state.current_player:
                return self.game_state.get_valid_actions(player_id)
            else:
                return []
        elif self.phase == GamePhase.RESPONSE:
            if player_id in self.response_deadline and self.pending_action:
                return self.game_state.get_response_actions(player_id, self.pending_action)
            else:
                return []
        elif self.phase == GamePhase.CHOICE:
            if player_id == self.waiting_for_choice:
                return self.choice_options
            else:
                return []
        else:
            return []
    
    def attempt_action(self, player_id: int, action: Action, target: Optional[int] = None) -> bool:
        """attempt to perform an action"""
        if self.phase != GamePhase.ACTION:
            return False
        
        if player_id != self.game_state.current_player:
            return False
        
        if action not in self.get_valid_actions(player_id):
            return False
        
        # create action data
        action_data = ActionData(action=action, actor=player_id, target=target)
        
        # check if action can be afforded
        if not self.game_state.players[player_id].can_afford(action_data.cost):
            return False
        
        # store the pending action
        self.pending_action = action_data
        
        # log the action attempt
        self.action_history.append({
            'type': 'action_attempt',
            'player': player_id,
            'action': action.value,
            'target': target,
            'turn': self.game_state.turn_count
        })
        
        # check if action can be responded to
        if self._action_can_be_responded_to(action_data):
            self._start_response_phase()
        else:
            # no responses possible, execute immediately
            self._execute_action()
        
        return True
    
    def submit_response(self, player_id: int, response: Action, claimed_role: Optional[Role] = None) -> bool:
        """submit a response to a pending action"""
        if self.phase != GamePhase.RESPONSE:
            return False
        
        if player_id not in self.response_deadline:
            return False
        
        if response not in self.get_valid_actions(player_id):
            return False
        
        # process the response
        if response == Action.PASS:
            self.pending_responses.append((player_id, ResponseType.PASS, None))
        elif response == Action.CHALLENGE_ACTION:
            self.pending_responses.append((player_id, ResponseType.CHALLENGE_ACTION, None))
        elif response == Action.CHALLENGE_BLOCK:
            self.pending_responses.append((player_id, ResponseType.CHALLENGE_BLOCK, None))
        elif response in [Action.BLOCK_FOREIGN_AID, Action.BLOCK_STEAL_CAPTAIN, 
                         Action.BLOCK_STEAL_AMBASSADOR, Action.BLOCK_ASSASSINATE]:
            # determine the role being claimed for the block
            if response == Action.BLOCK_FOREIGN_AID:
                claimed_role = Role.DUKE
            elif response == Action.BLOCK_STEAL_CAPTAIN:
                claimed_role = Role.CAPTAIN
            elif response == Action.BLOCK_STEAL_AMBASSADOR:
                claimed_role = Role.AMBASSADOR
            elif response == Action.BLOCK_ASSASSINATE:
                claimed_role = Role.CONTESSA
            
            self.pending_responses.append((player_id, ResponseType.BLOCK, claimed_role))
        
        # remove this player from response deadline
        self.response_deadline.discard(player_id)
        
        # log the response
        self.action_history.append({
            'type': 'response',
            'player': player_id,
            'response': response.value,
            'claimed_role': claimed_role.value if claimed_role else None,
            'turn': self.game_state.turn_count
        })
        
        # if all responses are in, process them
        if len(self.response_deadline) == 0:
            self._process_responses()
        
        return True
    
    def submit_choice(self, player_id: int, choice) -> bool:
        """submit a choice (like which card to lose)"""
        if self.phase != GamePhase.CHOICE:
            return False
        
        if player_id != self.waiting_for_choice:
            return False
        
        if choice not in self.choice_options:
            return False
        
        # process the choice based on type
        if self.choice_type == "lose_influence":
            # player chose which influence to lose
            role_to_lose = choice
            self.game_state.players[player_id].lose_influence(role_to_lose)
            
            # log the choice
            self.action_history.append({
                'type': 'influence_lost',
                'player': player_id,
                'role': role_to_lose.value,
                'turn': self.game_state.turn_count
            })
            
            # check if player is eliminated
            if self.game_state.players[player_id].is_eliminated:
                self.action_history.append({
                    'type': 'player_eliminated',
                    'player': player_id,
                    'turn': self.game_state.turn_count
                })
        
        elif self.choice_type == "exchange_cards":
            # player chose which cards to keep for exchange
            kept_cards = choice
            player = self.game_state.players[player_id]
            
            # put back the cards not kept
            all_cards = player.influence.copy()
            for card in all_cards:
                if card not in kept_cards:
                    player.influence.remove(card)
                    self.game_state.shuffle_role_back(card)
            
            # ensure player has exactly 2 cards
            while len(player.influence) < 2:
                new_card = self.game_state.draw_card()
                if new_card:
                    player.influence.append(new_card)
        
        # clear choice state
        self.waiting_for_choice = None
        self.choice_type = None
        self.choice_options = []
        
        # check if game is over
        if self.game_state.check_game_over():
            self.phase = GamePhase.EXECUTION  # end the game
            return True
        
        # continue with normal flow
        self._next_turn()
        return True
    
    def _action_can_be_responded_to(self, action_data: ActionData) -> bool:
        """check if an action can be responded to"""
        # actions that can be challenged
        challengeable_actions = [Action.TAX, Action.ASSASSINATE, Action.STEAL, Action.EXCHANGE]
        
        # actions that can be blocked
        blockable_actions = [Action.FOREIGN_AID, Action.STEAL, Action.ASSASSINATE]
        
        return (action_data.action in challengeable_actions or 
                action_data.action in blockable_actions)
    
    def _start_response_phase(self):
        """start the response phase"""
        self.phase = GamePhase.RESPONSE
        self.pending_responses = []
        self.response_deadline = set()
        
        # determine who can respond
        for player_id in range(self.game_state.num_players):
            if (not self.game_state.players[player_id].is_eliminated and
                self.game_state.can_player_respond(player_id, self.pending_action)):
                self.response_deadline.add(player_id)
    
    def _process_responses(self):
        """process all submitted responses"""
        # first check for challenges
        challenges = [r for r in self.pending_responses if r[1] == ResponseType.CHALLENGE_ACTION]
        
        if challenges:
            # process first challenge (first-come-first-serve)
            challenger_id, _, _ = challenges[0]
            self._resolve_challenge(challenger_id, self.pending_action)
            return
        
        # then check for blocks
        blocks = [r for r in self.pending_responses if r[1] == ResponseType.BLOCK]
        
        if blocks:
            # process first block
            blocker_id, _, claimed_role = blocks[0]
            self._resolve_block(blocker_id, claimed_role)
            return
        
        # no challenges or blocks, execute the action
        self._execute_action()
    
    def _resolve_challenge(self, challenger_id: int, action_data: ActionData):
        """resolve a challenge"""
        self.phase = GamePhase.CHALLENGE
        
        # use the game state's challenge resolution
        challenge_succeeded = self.game_state.resolve_challenge(challenger_id, action_data)
        
        # log the challenge result
        self.action_history.append({
            'type': 'challenge_result',
            'challenger': challenger_id,
            'actor': action_data.actor,
            'action': action_data.action.value,
            'succeeded': challenge_succeeded,
            'turn': self.game_state.turn_count
        })
        
        if challenge_succeeded:
            # challenge succeeded, action is canceled
            self.pending_action = None
            self._next_turn()
        else:
            # challenge failed, action proceeds
            self._execute_action()
    
    def _resolve_block(self, blocker_id: int, claimed_role: Role):
        """resolve a block"""
        self.phase = GamePhase.BLOCK
        
        # check if anyone wants to challenge the block
        # for now, auto-proceed without block challenges
        # todo: implement block challenge phase
        
        # block succeeds, action is canceled
        self.action_history.append({
            'type': 'block_success',
            'blocker': blocker_id,
            'claimed_role': claimed_role.value,
            'blocked_action': self.pending_action.action.value,
            'turn': self.game_state.turn_count
        })
        
        self.pending_action = None
        self._next_turn()
    
    def _execute_action(self):
        """execute the pending action"""
        if not self.pending_action:
            return
        
        self.phase = GamePhase.EXECUTION
        
        # special handling for actions that require choices
        if self.pending_action.action == Action.EXCHANGE:
            self._handle_exchange_action()
        elif self.pending_action.action in [Action.COUP, Action.ASSASSINATE]:
            self._handle_targeted_elimination()
        else:
            # execute the action normally
            self.game_state.execute_action(self.pending_action)
            self._log_action_execution()
            self.pending_action = None
            self._next_turn()
    
    def _handle_exchange_action(self):
        """handle exchange action which requires player choice"""
        player_id = self.pending_action.actor
        player = self.game_state.players[player_id]
        
        # draw 2 cards
        drawn_cards = []
        for _ in range(2):
            card = self.game_state.draw_card()
            if card:
                drawn_cards.append(card)
                player.influence.append(card)
        
        # now player needs to choose which cards to keep
        self.phase = GamePhase.CHOICE
        self.waiting_for_choice = player_id
        self.choice_type = "exchange_cards"
        self.choice_options = player.influence.copy()  # all current cards
        
        # for now, auto-choose to keep original 2 cards (simplified)
        # todo: implement proper choice mechanism
        cards_to_keep = player.influence[:2]
        self.submit_choice(player_id, cards_to_keep)
    
    def _handle_targeted_elimination(self):
        """handle coup/assassinate which requires target to choose influence to lose"""
        if self.pending_action.target is None:
            self.pending_action = None
            self._next_turn()
            return
        
        target_id = self.pending_action.target
        target = self.game_state.players[target_id]
        
        # pay the cost first
        self.game_state.players[self.pending_action.actor].coins -= self.pending_action.cost
        
        if not target.influence:
            # target has no influence to lose
            self.pending_action = None
            self._next_turn()
            return
        
        # target needs to choose which influence to lose
        self.phase = GamePhase.CHOICE
        self.waiting_for_choice = target_id
        self.choice_type = "lose_influence"
        self.choice_options = target.influence.copy()
        
        # for now, auto-choose first influence (simplified)
        # todo: implement proper choice mechanism
        influence_to_lose = target.influence[0]
        self.submit_choice(target_id, influence_to_lose)
    
    def _log_action_execution(self):
        """log the execution of an action"""
        self.action_history.append({
            'type': 'action_executed',
            'player': self.pending_action.actor,
            'action': self.pending_action.action.value,
            'target': self.pending_action.target,
            'turn': self.game_state.turn_count
        })
    
    def _next_turn(self):
        """advance to the next turn"""
        self.game_state.turn_count += 1
        
        # check if game is over
        if self.game_state.check_game_over():
            self.phase = GamePhase.EXECUTION  # game over
            return
        
        # advance to next player
        self.game_state.next_player()
        self.phase = GamePhase.ACTION
        
        # clear any pending state
        self.pending_action = None
        self.pending_responses = []
        self.response_deadline = set()
    
    def get_game_state_copy(self) -> GameState:
        """get a copy of the current game state"""
        return self.game_state.copy()
    
    def get_action_history(self) -> List[Dict]:
        """get the action history"""
        return self.action_history.copy()
    
    def reset(self, seed: Optional[int] = None) -> GameState:
        """reset the game to initial state"""
        if seed is not None:
            random.seed(seed)
        
        self.game_state = GameState(self.game_state.num_players)
        self.phase = GamePhase.ACTION
        self.pending_action = None
        self.pending_responses = []
        self.response_deadline = set()
        self.waiting_for_choice = None
        self.choice_type = None
        self.choice_options = []
        self.action_history = []
        self.turn_history = []
        
        return self.game_state.copy() 