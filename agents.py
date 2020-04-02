import keyboard
import numpy as np
from environment import Player, GameAction, GameState, get_next_state

class KeyboardPlayer(Player):
    KEY_PRESSED = GameAction.STRAIGHT
    
    def __init__(self, use_keyboard_listener=False):
        """
        
        :param use_keyboard_listener: If you're using Mac operation system, you MUST use 'False' for this to work!
        """
        super().__init__()
        self.use_keyboard_listener = use_keyboard_listener
        if use_keyboard_listener:
            keyboard.add_hotkey('a', KeyboardPlayer.turn_left)
            keyboard.add_hotkey('w', KeyboardPlayer.go_straight)
            keyboard.add_hotkey('d', KeyboardPlayer.turn_right)
    
    @staticmethod
    def turn_left():
        KeyboardPlayer.KEY_PRESSED = GameAction.LEFT
    
    @staticmethod
    def go_straight():
        KeyboardPlayer.KEY_PRESSED = GameAction.STRAIGHT
    
    @staticmethod
    def turn_right():
        KeyboardPlayer.KEY_PRESSED = GameAction.RIGHT
        
    def get_action(self, state: GameState) -> GameAction:
        if not self.use_keyboard_listener:
            left_key, right_key = 'a', 'd'
            a = input(f"Enter your move ({left_key} => left / {right_key} => right / ENTER => straight):")
            if len(a) != 1 or a not in [left_key, right_key]:
                a = GameAction.STRAIGHT
            else:
                a = GameAction.LEFT if a == left_key else GameAction.RIGHT
        else:
            a = KeyboardPlayer.KEY_PRESSED
            self.go_straight()
        return a


class RandomPlayer(Player):
    def get_action(self, state: GameState) -> GameAction:
        i = np.random.randint(low=0, high=3)
        return list(GameAction)[i]


class StaticAgent(Player):
    def __init__(self, actions: tuple):
        super().__init__()
        self._actions = actions
        self._current_action = 0

    def get_action(self, state: GameState) -> GameAction:
        action_index = self._current_action
        self._current_action += 1
        return self._actions[action_index] if action_index < len(self._actions) else self._actions[-1]


class GreedyAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        # init with all possible actions for the case where the agent is alone. it will (possibly) be overridden later
        best_actions = state.get_possible_actions(player_index=self.player_index)
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            for opponents_actions in state.get_possible_actions_dicts_given_action(action, player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                h_value = self._heuristic(next_state)
                if h_value > best_value:
                    best_value = h_value
                    best_actions = [action]
                elif h_value == best_value:
                    best_actions.append(action)

                if len(state.opponents_alive) > 2:
                    # consider only 1 possible opponents actions to reduce time & memory:
                    break
        return np.random.choice(best_actions)

    def _heuristic(self, state: GameState) -> float:
        if not state.snakes[self.player_index].alive:
            return state.snakes[self.player_index].length
        discount_factor = 0.5
        """
        reward for each fruit eaten = 1 since the snake is elongated by 1.
        We want to calculate an optimistic reward value for the optimistic scenario that we eat a fruit in every step
        until the game ends (or until the fruits end). don't confuse the term optimistic with 'admissible', which means 
        always over-estimating the value. Here by optimistic we refer to the fact that the agent believes he can eat 
        all the fruits. 
        thus, if we discount each step into the future (since it has less value because it's less certain) we get:
        denote discount_factor = a
        k = the number of steps left in the game (until the game ends).
        then:
        optimistic sum of rewards = (a*1) + (a^2)*1 + (a^3)*1 +... + (a^k)*1
        = a + a^2 + a^3 +... + a^k  (multiplying by 1 disappears)
        = a*( 1-(a^k) ) / (1-a)       (sum of geometric series)
        
        """
        max_possible_fruits = len(state.fruits_locations) + sum([s.length for s in state.snakes
                                                                 if s.index != self.player_index and s.alive])
        turns_left = (state.game_duration_in_turns - state.turn_number)
        max_possible_fruits = min(max_possible_fruits, turns_left)
        optimistic_future_reward = discount_factor*(1 - discount_factor ** max_possible_fruits) / (1-discount_factor)
        return state.snakes[self.player_index].length + optimistic_future_reward


class BetterGreedyAgent(GreedyAgent):
    def _heuristic(self, state: GameState) -> float:
        from submission import heuristic
        return heuristic(state, self.player_index)
