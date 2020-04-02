from environment import Player, GameState, GameAction, get_next_state
from utils import get_fitness
import numpy as np
from enum import Enum
from scipy.spatial.distance import cityblock
from agents import RandomPlayer


def heuristic(state: GameState, player_index: int) -> float:
    """
    Computes the heuristic value for the agent with player_index at the given state
    :param state:
    :param player_index: integer. represents the identity of the player. this is the index of the agent's snake in the
    state.snakes array as well.
    :return:
    """

    "lowest as possible in order to stay alive"
    if not state.snakes[player_index].alive:
        return state.snakes[player_index].length * 0.002

    "eating all the fruits is one of our goals"
    if len(state.fruits_locations) == 0:
        return state.board_size.height * state.board_size.width * state.snakes[player_index].length

    "eat as much as you can"
    ate_this_turn = (state.snakes[player_index].old_tail_position == state.snakes[player_index].tail_position)

    "calculate the shortest manhattan distance from agent's head to opp's head or tail"
    opp_manhattan_dists = min([min(cityblock(opp.head, state.snakes[player_index].head),
                                   cityblock(opp.tail_position, state.snakes[player_index].head))
                               for opp in state.snakes if opp.index != player_index])

    "calculate the shortest manhattan distance from agent's head to the nearest fruit"
    fruit_manhattan_dists = min(cityblock(fruit, state.snakes[player_index].head)
                                for fruit in state.fruits_locations)

    return 0.4 * ate_this_turn + (state.board_size.height * state.board_size.width) * state.snakes[
        player_index].length + 0.1 * opp_manhattan_dists - 0.4 * fruit_manhattan_dists - \
        0.4 * len(state.fruits_locations)


class MinimaxAgent(Player):
    """
    This class implements the Minimax algorithm.
    Since this algorithm needs the game to have defined turns, we will model these turns ourselves.
    Use 'TurnBasedGameState' to wrap the given state at the 'get_action' method.
    hint: use the 'agent_action' property to determine if it's the agents turn or the opponents' turn. You can pass
    'None' value (without quotes) to indicate that your agent haven't picked an action yet.
    """

    class Turn(Enum):
        AGENT_TURN = 'AGENT_TURN'
        OPPONENTS_TURN = 'OPPONENTS_TURN'

    class TurnBasedGameState:
        """
        This class is a wrapper class for a GameState. It holds the action of our agent as well, so we can model turns
        in the game (set agent_action=None to indicate that our agent has yet to pick an action).
        """

        def __init__(self, game_state: GameState, agent_action: GameAction):
            self.game_state = game_state
            self.agent_action = agent_action

        @property
        def turn(self):
            return MinimaxAgent.Turn.AGENT_TURN if self.agent_action is None else MinimaxAgent.Turn.OPPONENTS_TURN

    def get_action(self, state: GameState) -> GameAction:
        max_value = np.NINF
        best_action = None
        for action in state.get_possible_actions(self.player_index):
            value = self.rb_minimax(self.TurnBasedGameState(state, action), 2)
            max_value = max(max_value, value)
            best_action = action if max_value == value else best_action
        return best_action

    def rb_minimax(self, state: TurnBasedGameState, depth: int):
        if state.game_state.turn_number == state.game_state.game_duration_in_turns:
            if state.game_state.current_winner == self.player_index:
                return state.game_state.snakes[self.player_index].length ** 2
            else:
                return -1

        if len(state.game_state.living_agents) == 0:
            return -1

        if depth == 0:
            return heuristic(state.game_state, self.player_index)

        if state.turn == self.Turn.AGENT_TURN:
            current_max = np.NINF
            for action in state.game_state.get_possible_actions(self.player_index):
                value = self.rb_minimax(self.TurnBasedGameState(state.game_state, action), depth)
                current_max = max(current_max, value)
            return current_max
        else:
            current_min = np.Inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                value = self.rb_minimax(self.TurnBasedGameState(next_state, None), depth - 1)
                current_min = min(current_min, value)
            return current_min


class AlphaBetaAgent(MinimaxAgent):
    def get_action(self, state: GameState) -> GameAction:
        max_value = np.NINF
        best_action = None
        for action in state.get_possible_actions(self.player_index):
            value = self.rb_alpha_beta(self.TurnBasedGameState(state, action), 2, np.NINF, np.Inf)
            max_value = max(max_value, value)
            best_action = action if max_value == value else best_action
        return best_action

    def rb_alpha_beta(self, state: MinimaxAgent.TurnBasedGameState, depth: int, alpha: float, beta: float):
        if state.game_state.turn_number == state.game_state.game_duration_in_turns:
            if state.game_state.current_winner == self.player_index:
                return state.game_state.snakes[self.player_index].length ** 2
            else:
                return -1

        if len(state.game_state.living_agents) == 0:
            return -1

        if depth == 0:
            return heuristic(state.game_state, self.player_index)

        if state.turn == self.Turn.AGENT_TURN:
            current_max = np.NINF
            for action in state.game_state.get_possible_actions(self.player_index):
                value = self.rb_alpha_beta(self.TurnBasedGameState(state.game_state, action), depth, alpha, beta)
                current_max = max(current_max, value)
                alpha = max(current_max, alpha)
                if current_max >= beta:
                    return np.Inf
            return current_max
        else:
            current_min = np.Inf
            for opponents_actions in state.game_state.get_possible_actions_dicts_given_action(state.agent_action,
                                                                                              player_index=self.player_index):
                opponents_actions[self.player_index] = state.agent_action
                next_state = get_next_state(state.game_state, opponents_actions)
                value = self.rb_alpha_beta(self.TurnBasedGameState(next_state, None), depth - 1, alpha, beta)
                beta = min(current_min, beta)
                current_min = min(current_min, value)
                if current_min <= alpha:
                    return np.NINF
            return current_min


def SAHC_sideways():
    """
    Implement Steepest Ascent Hill Climbing with Sideways Steps Here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    n = 50

    "initial state derived from several experiments as described in the PDF"
    current_state = [GameAction.LEFT, GameAction.STRAIGHT, GameAction.RIGHT, GameAction.RIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.RIGHT, GameAction.STRAIGHT, GameAction.LEFT,
                     GameAction.RIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.RIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.RIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.LEFT, GameAction.RIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,
                     GameAction.STRAIGHT, GameAction.STRAIGHT, GameAction.STRAIGHT,  GameAction.RIGHT, GameAction.LEFT,
                     GameAction.LEFT]

    best_val = np.NINF
    for i in range(n):
        best_states = []
        for action in list(GameAction):
            new_state = current_state.copy()
            new_state[i] = action
            new_value = get_fitness(tuple(new_state))
            if new_value > best_val:
                best_val = new_value
                best_states = [new_state]
            elif new_value == best_val:
                best_states += [new_state]
        random_index = np.random.choice(range(len(best_states)))
        current_state = best_states[random_index]
    print(current_state)


def local_search():
    """
    Implement your own local search algorithm here.
    We give you the freedom to choose an initial state as you wish. You may start with a deterministic state (think of
    examples, what interesting options do you have?), or you may randomly sample one (you may use any distribution you
    like). In any case, write it in your report and describe your choice.

    an outline of the algorithm can be
    1) pick an initial state/states
    2) perform the search according to the algorithm
    3) print the best moves vector you found.
    :return:
    """
    k = 15
    n = 50
    list_new_beam = [[np.random.choice(list(GameAction), p=[0.05, 0.9, 0.05]) for _ in range(n)] for _ in range(k)]
    new_beam = dict()
    for item in list_new_beam:
        new_beam[get_fitness(tuple(item))] = item
    for i in range(n):
        beam = new_beam.copy()
        new_beam = dict()
        for s in beam.values():
            for action in list(GameAction):
                new = s.copy()
                new[i] = action
                if len(new_beam) < k:
                    new_beam[get_fitness(tuple(new))] = new
                else:
                    h_new = get_fitness(tuple(new))
                    if h_new > min(new_beam.keys()):
                        new_beam.pop(min(new_beam.keys()))
                        new_beam[h_new] = new
    print(new_beam[max(new_beam.keys())])
    print(get_fitness(tuple(new_beam[max(new_beam.keys())])))


class TournamentAgent(Player):

    def get_action(self, state: GameState) -> GameAction:
        # init with all possible actions for the case where the agent is alone. it will (possibly) be overridden later
        best_actions = state.get_possible_actions(player_index=self.player_index)
        best_value = -np.inf
        for action in state.get_possible_actions(player_index=self.player_index):
            for opponents_actions in state.get_possible_actions_dicts_given_action(action,
                                                                                   player_index=self.player_index):
                opponents_actions[self.player_index] = action
                next_state = get_next_state(state, opponents_actions)
                h_value = heuristic(next_state, self.player_index)
                if h_value > best_value:
                    best_value = h_value
                    best_actions = [action]
                elif h_value == best_value:
                    best_actions.append(action)

                if len(state.opponents_alive) > 2:
                    # consider only 1 possible opponents actions to reduce time & memory:
                    break
        return np.random.choice(best_actions)


if __name__ == '__main__':
    SAHC_sideways()
    local_search()
