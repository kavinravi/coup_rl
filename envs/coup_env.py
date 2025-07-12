import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from coup_controller import CoupController, GamePhase
from coup_game import Action, Role, GameState
import random


class CoupEnv(gym.Env):
    """gymnasium environment for coup game"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}
    
    def __init__(self, num_players: int = 2, max_turns: int = 100, render_mode: Optional[str] = None):
        super().__init__()
        
        self.num_players = num_players
        self.max_turns = max_turns
        self.render_mode = render_mode
        
        # initialize controller
        self.controller = CoupController(num_players=num_players)
        
        # action space - all possible actions
        self.action_space = spaces.Discrete(len(Action))
        
        # observation space
        # this will be a dict with:
        # - public info (coins, lost influence, etc.)
        # - private info (own cards)
        # - game state (current player, phase, etc.)
        # - action history buffer
        self.observation_space = spaces.Dict({
            # public game state
            "coins": spaces.Box(low=0, high=100, shape=(num_players,), dtype=np.int32),
            "lost_influence": spaces.Box(low=0, high=2, shape=(num_players, 5), dtype=np.int32),  # 5 roles
            "is_eliminated": spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.int32),
            
            # private state (only for current player)
            "hand": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int32),  # which roles I have
            
            # game state
            "current_player": spaces.Discrete(num_players),
            "phase": spaces.Discrete(len(GamePhase)),
            "turn_count": spaces.Box(low=0, high=max_turns, shape=(), dtype=np.int32),
            
            # action context
            "pending_action": spaces.Box(low=0, high=len(Action), shape=(), dtype=np.int32),
            "pending_target": spaces.Box(low=-1, high=num_players-1, shape=(), dtype=np.int32),  # -1 for no target
            "can_respond": spaces.Box(low=0, high=1, shape=(), dtype=np.int32),
            
            # action history buffer (last 20 actions)
            "action_history": spaces.Box(low=0, high=len(Action), shape=(20,), dtype=np.int32),
            "action_players": spaces.Box(low=0, high=num_players, shape=(20,), dtype=np.int32),
            "action_targets": spaces.Box(low=-1, high=num_players-1, shape=(20,), dtype=np.int32),
            
            # valid actions mask
            "valid_actions": spaces.Box(low=0, high=1, shape=(len(Action),), dtype=np.int32),
        })
        
        # action mappings
        self.action_list = list(Action)
        self.action_to_int = {action: i for i, action in enumerate(self.action_list)}
        self.int_to_action = {i: action for i, action in enumerate(self.action_list)}
        
        # role mappings
        self.role_list = list(Role)
        self.role_to_int = {role: i for i, role in enumerate(self.role_list)}
        
        # phase mappings
        self.phase_list = list(GamePhase)
        self.phase_to_int = {phase: i for i, phase in enumerate(self.phase_list)}
        
        # episode tracking
        self.episode_rewards = [0.0] * num_players
        self.episode_step_count = 0
        
        # history buffer
        self.action_history_buffer = []
        self.max_history_length = 20
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """reset the environment"""
        super().reset(seed=seed)
        
        # reset controller
        self.controller.reset(seed=seed)
        
        # reset episode tracking
        self.episode_rewards = [0.0] * self.num_players
        self.episode_step_count = 0
        self.action_history_buffer = []
        
        # get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """step the environment"""
        self.episode_step_count += 1
        
        # decode action
        if action not in self.int_to_action:
            return self._get_observation(), -1.0, False, False, {"error": "invalid action"}
        
        action_enum = self.int_to_action[action]
        
        # determine current player
        current_player = self.controller.get_current_player()
        
        # attempt the action
        success = self._attempt_action(current_player, action_enum)
        
        # calculate reward
        reward = self._calculate_reward(current_player, action_enum, success)
        self.episode_rewards[current_player] += reward
        
        # check if game is over
        terminated = self.controller.is_game_over()
        truncated = self.episode_step_count >= self.max_turns
        
        # get observation and info
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _attempt_action(self, player_id: int, action: Action) -> bool:
        """attempt to perform an action"""
        if self.controller.get_current_phase() == GamePhase.ACTION:
            # primary action
            target = self._get_action_target(player_id, action)
            return self.controller.attempt_action(player_id, action, target)
        
        elif self.controller.get_current_phase() == GamePhase.RESPONSE:
            # response to pending action
            return self.controller.submit_response(player_id, action)
        
        elif self.controller.get_current_phase() == GamePhase.CHOICE:
            # choice (like which influence to lose)
            choice_options = self.controller.choice_options
            if len(choice_options) > 0:
                # for now, just pick the first available choice
                # todo: make this more sophisticated
                return self.controller.submit_choice(player_id, choice_options[0])
        
        return False
    
    def _get_action_target(self, player_id: int, action: Action) -> Optional[int]:
        """get the target for an action that requires one"""
        if action not in [Action.COUP, Action.ASSASSINATE, Action.STEAL]:
            return None
        
        # find valid targets
        valid_targets = []
        for i in range(self.num_players):
            if i != player_id and not self.controller.game_state.players[i].is_eliminated:
                valid_targets.append(i)
        
        if len(valid_targets) == 0:
            return None
        
        # for now, just pick the first valid target
        # todo: make this more sophisticated or let agent choose
        return valid_targets[0]
    
    def _calculate_reward(self, player_id: int, action: Action, success: bool) -> float:
        """calculate reward for an action"""
        if not success:
            return -0.1  # small penalty for invalid actions
        
        # base reward components
        reward = 0.0
        
        # small per-turn penalty to encourage ending games
        reward -= 0.01
        
        # reward for successful actions
        if action == Action.INCOME:
            reward += 0.05
        elif action == Action.FOREIGN_AID:
            reward += 0.1
        elif action == Action.TAX:
            reward += 0.15
        elif action == Action.STEAL:
            reward += 0.2
        elif action == Action.ASSASSINATE:
            reward += 0.3
        elif action == Action.COUP:
            reward += 0.25
        
        # reward for eliminating opponents
        current_active = len(self.controller.get_active_players())
        if current_active < self.num_players:
            reward += 0.5  # bonus for eliminating someone
        
        # terminal rewards
        if self.controller.is_game_over():
            winner = self.controller.get_winner()
            if winner == player_id:
                reward += 10.0  # large reward for winning
            else:
                reward -= 5.0   # penalty for losing
        
        return reward
    
    def _get_observation(self) -> Dict:
        """get current observation"""
        game_state = self.controller.game_state
        current_player = self.controller.get_current_player()
        
        # during response phase, "current player" is the first player who can respond
        if self.controller.get_current_phase() == GamePhase.RESPONSE:
            if len(self.controller.response_deadline) > 0:
                current_player = min(self.controller.response_deadline)
        
        # public information
        coins = np.array([p.coins for p in game_state.players], dtype=np.int32)
        
        # lost influence matrix (players x roles)
        lost_influence = np.zeros((self.num_players, 5), dtype=np.int32)
        for i, player in enumerate(game_state.players):
            for role in player.lost_influence:
                role_idx = self.role_to_int[role]
                lost_influence[i][role_idx] += 1
        
        eliminated = np.array([int(p.is_eliminated) for p in game_state.players], dtype=np.int32)
        
        # private information (current player's hand)
        hand = np.zeros(5, dtype=np.int32)
        if current_player < len(game_state.players):
            for role in game_state.players[current_player].influence:
                role_idx = self.role_to_int[role]
                hand[role_idx] = 1
        
        # game state
        phase = self.phase_to_int[self.controller.get_current_phase()]
        
        # pending action info
        pending_action = 0
        pending_target = -1
        if self.controller.pending_action:
            pending_action = self.action_to_int[self.controller.pending_action.action]
            pending_target = self.controller.pending_action.target if self.controller.pending_action.target is not None else -1
        
        # can respond flag
        can_respond = 0
        if self.controller.get_current_phase() == GamePhase.RESPONSE:
            can_respond = 1
        
        # action history
        history_actions = np.zeros(self.max_history_length, dtype=np.int32)
        history_players = np.zeros(self.max_history_length, dtype=np.int32)
        history_targets = np.full(self.max_history_length, -1, dtype=np.int32)
        
        # fill history buffer
        history = self.controller.get_action_history()
        for i, event in enumerate(history[-self.max_history_length:]):
            if event.get('type') == 'action_attempt':
                history_actions[i] = self.action_to_int.get(Action(event['action']), 0)
                history_players[i] = event['player']
                target = event.get('target', -1)
                history_targets[i] = target if target is not None else -1
        
        # valid actions mask
        valid_actions = np.zeros(len(Action), dtype=np.int32)
        for action in self.controller.get_valid_actions(current_player):
            valid_actions[self.action_to_int[action]] = 1
        
        return {
            "coins": coins,
            "lost_influence": lost_influence,
            "is_eliminated": eliminated,
            "hand": hand,
            "current_player": current_player,
            "phase": phase,
            "turn_count": np.int32(game_state.turn_count),
            "pending_action": np.int32(pending_action),
            "pending_target": np.int32(pending_target),
            "can_respond": np.int32(can_respond),
            "action_history": history_actions,
            "action_players": history_players,
            "action_targets": history_targets,
            "valid_actions": valid_actions,
        }
    
    def _get_info(self) -> Dict:
        """get additional info"""
        return {
            "current_player": self.controller.get_current_player(),
            "phase": self.controller.get_current_phase().value,
            "active_players": self.controller.get_active_players(),
            "game_over": self.controller.is_game_over(),
            "winner": self.controller.get_winner(),
            "episode_rewards": self.episode_rewards.copy(),
            "episode_length": self.episode_step_count,
        }
    
    def render(self):
        """render the game state"""
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """render in human-readable format"""
        game_state = self.controller.game_state
        
        print(f"\n=== Coup Game - Turn {game_state.turn_count} ===")
        print(f"Phase: {self.controller.get_current_phase().value}")
        print(f"Current Player: {self.controller.get_current_player()}")
        
        # player info
        for i, player in enumerate(game_state.players):
            status = "ELIMINATED" if player.is_eliminated else "ACTIVE"
            influence_count = len(player.influence)
            lost_roles = [r.value for r in player.lost_influence]
            
            print(f"Player {i}: {player.coins} coins, {influence_count} influence, {status}")
            if lost_roles:
                print(f"  Lost: {', '.join(lost_roles)}")
        
        # pending action
        if self.controller.pending_action:
            action = self.controller.pending_action
            print(f"Pending: {action.action.value} by Player {action.actor}")
            if action.target is not None:
                print(f"  Target: Player {action.target}")
        
        # valid actions for current player
        current_player = self.controller.get_current_player()
        valid_actions = self.controller.get_valid_actions(current_player)
        if valid_actions:
            print(f"Valid actions for Player {current_player}: {[a.value for a in valid_actions]}")
        
        print("=" * 40)
    
    def _render_rgb_array(self) -> np.ndarray:
        """render as rgb array (placeholder)"""
        # todo: implement visual rendering
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """close the environment"""
        pass


# utility functions for multi-agent scenarios
def make_coup_env(num_players: int = 2, **kwargs) -> CoupEnv:
    """create a coup environment"""
    return CoupEnv(num_players=num_players, **kwargs)


def get_observation_space(num_players: int = 2) -> spaces.Dict:
    """get the observation space for a given number of players"""
    env = CoupEnv(num_players=num_players)
    return env.observation_space


def get_action_space() -> spaces.Discrete:
    """get the action space"""
    return spaces.Discrete(len(Action)) 