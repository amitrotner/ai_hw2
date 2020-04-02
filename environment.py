import numpy as np
from scipy.spatial.distance import cityblock
from enum import Enum
from collections import namedtuple
from typing import List
from PIL import Image
from gym.envs.classic_control import rendering
import logging
import time
import copy
from abc import ABC, abstractmethod

class GameAction(Enum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


Grid2DSize = namedtuple('Grid2DSize', ['width', 'height'])


class GridTooSmallException(Exception):
    pass


class GameLobbyIsFullException(Exception):
    pass


class AgentOps(Enum):
    RESET = 0
    DO_ACTION = 1


GameOp = namedtuple('GameOp', ['playerIndex', 'op', 'args'])


WinnerAtTurn = namedtuple("WinnerAtTurn", ['player_index', 'length'])
WinnerAtTurnList = List[WinnerAtTurn]


class SnakeMovementDirections(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class SnakeAgent:
    """
    This class represents the snake figure in the game's grid. This is NOT the controller, just the entity
    (data structure) responsible for handling each snake on the board.
    """
    REWARD_FOR_BEING_ALIVE = 0
    REWARD_FOR_TROPHY = 1
    REWARD_FOR_DEATH = 0

    def __init__(self, player_index: int, initial_head_position: tuple):
        self._player_index = player_index
        self.direction = self._pick_initial_direction()
        self.position = self.initialize_position(initial_head_position)
        self._eaten_trophies_positions = []
        self._next_action = None
        self._alive = True
        self.reward = self.REWARD_FOR_BEING_ALIVE

        # This list will be used by the backend to update the map:
        self.old_tail_position = None

    @property
    def index(self):
        return self._player_index

    @property
    def next_action(self) -> GameAction:
        return self._next_action

    @next_action.setter
    def next_action(self, value: GameAction):
        self._next_action = value

    @property
    def head(self):
        """
        Return the position of the head of the snake.
        :return: tuple (row: int, col: int) corresponding to the position of the snake's head.
        """
        return self.position[-1]

    @property
    def tail_position(self):
        return self.position[0]

    @property
    def alive(self):
        return self._alive

    @property
    def length(self):
        return len(self.position)

    @staticmethod
    def _pick_initial_direction():
        possible_directions = list(SnakeMovementDirections)
        return possible_directions[np.random.randint(0, len(possible_directions))]

    def initialize_position(self, initial_head_position: tuple):
        """
        Given the (x, y) coordinates of the head generated for the snake agent, generate the coordinates list for the
        whole body of the snake agent. Consider the direction generated, and assume there is enough room for the body.
        Assume initial length of the snake = 2.
        Always assume the last element in this array represents the head of the snake.
        :param initial_head_position: tuple (x,y).
        :return: list of tuples [(x,y), ..., (x,y)]
        """
        body_positions = []
        if self.direction == SnakeMovementDirections.UP:
            body_positions.append((initial_head_position[0] + 1, initial_head_position[1]))
        if self.direction == SnakeMovementDirections.DOWN:
            body_positions.append((initial_head_position[0] - 1, initial_head_position[1]))
        if self.direction == SnakeMovementDirections.LEFT:
            body_positions.append((initial_head_position[0], initial_head_position[1] + 1))
        if self.direction == SnakeMovementDirections.RIGHT:
            body_positions.append((initial_head_position[0], initial_head_position[1] - 1))

        body_positions.append(initial_head_position)

        return body_positions

    def is_in_cell(self, cell_position):
        for pos in self.position:
            if pos == cell_position:
                return True
        return False

    def perform_action(self):
        """
        perform the action stored in the 'next_action' buffer. This only adds the new head location to the position
        array of the snake and handles the tail position as well (deletes the last place if no trophy was eaten at
        that location).
        :return:
        """
        assert self._next_action is not None
        assert self.alive

        self.reward = self.REWARD_FOR_BEING_ALIVE

        self._update_direction()
        old_head_pos = self.head

        # Compute new head position:
        if self.direction == SnakeMovementDirections.UP:
            new_head_pos = (old_head_pos[0] - 1, old_head_pos[1])
        elif self.direction == SnakeMovementDirections.DOWN:
            new_head_pos = (old_head_pos[0] + 1, old_head_pos[1])
        elif self.direction == SnakeMovementDirections.LEFT:
            new_head_pos = (old_head_pos[0], old_head_pos[1] - 1)
        else:
            assert self.direction == SnakeMovementDirections.RIGHT
            new_head_pos = (old_head_pos[0], old_head_pos[1] + 1)

        self.position.append(new_head_pos)

        # Handle tail:
        # In this version of the game, the growth after eating a fruit is instantaneous!
        self.old_tail_position = self.tail_position
        # self.pop_tail()  Not yet. Only if a fruit wasn't eaten

        self._next_action = None

    def pop_tail(self):
        """
        Remove the last tail position
        :return:
        """
        self.position.pop(0)

    def _update_direction(self):
        """
        Updates the direction of the snake based on the nextAction picked by the agent.
        :return: None
        """
        directions = list(SnakeMovementDirections)
        new_direction_idx = (self.direction.value + (self.next_action.value - 1)) % 4
        self.direction = directions[new_direction_idx]

    def kill(self):
        self.reward = self.REWARD_FOR_DEATH
        self._alive = False

    def eat_trophy(self, trophyPosition):
        self.reward += self.REWARD_FOR_TROPHY
        self._eaten_trophies_positions.append(trophyPosition)


SnakeAgentsList = List[SnakeAgent]


class GameState:
    FRUIT_VALUE = -1

    def __init__(
            self,
            turn_number: int,
            game_duration_in_turns: int,
            board_size: Grid2DSize,
            current_winner: WinnerAtTurn,
            fruits_locations: list,
            snakes: SnakeAgentsList):
        self.turn_number = turn_number
        self.game_duration_in_turns = game_duration_in_turns
        self.board_size = board_size
        self.current_winner: WinnerAtTurn = current_winner

        self.fruits_locations = copy.deepcopy(fruits_locations)
        self.snakes: SnakeAgentsList = copy.deepcopy(snakes)
        self.grid_map = self._build_grid_map(self.snakes, self.fruits_locations)

    def get_possible_actions(self, player_index=0) -> list:
        """
        get the possible actions of agent with player_index, in the current state of the game.
        :return:
        """
        if self.turn_number >= self.game_duration_in_turns or not self.snakes[player_index].alive:
            return []
        return list(GameAction)

    def get_possible_actions_dicts_given_action(self, action: GameAction, player_index=0) -> list:
        """
        Given the current player's action, return a list of all possible action dictionaries in which the current player
        chooses the given action. This method takes into account dead snakes and thus the result is not necessarily
        3 possible actions for each other player.
        Also, if the given state is terminal, an empty list will be returned.
        :param action: action of the player (non-changing).
        :param player_index: the index of the player who's action is given.
        :return: a list of dictionaries. each dictionary has the form {player_index ==> player_action}, i.e maps between
        living players and their moves. the list includes a dict for each possible move of the opponents.
        """
        assert self.snakes[player_index].alive

        opponents_alive = self.get_opponents_alive(player_index=player_index)
        if len(opponents_alive) == 0:
            yield {player_index: action}
        else:
            for i in range(len(GameAction) ** len(opponents_alive)):
                opponents_actions_str = np.base_repr(i, base=len(GameAction))
                opponents_actions_str = '0'*(len(opponents_alive) - len(opponents_actions_str)) + opponents_actions_str
                # print(opponents_actions_str)
                snake_actions = list(GameAction)
                possible_actions_dict = {opp: snake_actions[int(opp_action_str)]
                                         for opp, opp_action_str in zip(opponents_alive, opponents_actions_str)}
                possible_actions_dict[player_index] = action
                yield possible_actions_dict

    @property
    def n_agents(self):
        return len(self.snakes)

    @property
    def living_agents(self):
        """
        Return a list of indices of the living agents.
        :return: list of integers representing the indices of living agents in the current game.
        """
        return [i for i in range(self.n_agents) if self.snakes[i].alive]

    @property
    def opponents_alive(self):
        """
        Shortcut for getting the opponents of the first agent (usually, your agent)
        :return:
        """
        return self.get_opponents_alive(player_index=0)

    def get_opponents_alive(self, player_index=0):
        return [i for i in range(self.n_agents) if self.snakes[i].alive and i != player_index]

    @property
    def is_terminal_state(self):
        if self.turn_number >= self.game_duration_in_turns:
            return True
        if any([snake.alive for snake in self.snakes]):
            return False
        return True

    def get_board(self, player_perspective=0) -> np.ndarray:
        """
        Computes the state for a specific agent.
        The state is composed of n_snakes+1 layers / matrices (each has the shape of the board).
        The agent's snake (indicated by player_index) will be in the first layer (matrix),
        then the rest n_snakes-1 opponents' snakes,
        and in the last layer (matrix) of the tensor is the placements of the fruits.

        :param player_perspective: int. the index of the agent.
        :return: numpy array with the shape defined in the observation space.
        """
        state = self._get_empty_board()

        # Create a map between agent index and it's layer index in the state tensor:
        agent_to_layer_map = {idx: idx + 1 if idx < player_perspective else idx for idx in range(self.n_agents)}
        agent_to_layer_map[player_perspective] = 0

        # Show all living agents (snakes):
        color = 1.0
        for i, agent in enumerate(self.snakes):
            if not agent.alive:
                continue

            for pos_i, pos in enumerate(reversed(agent.position)):
                state[agent_to_layer_map[i], pos[0], pos[1]] = pos_i + 1

        # Show trophies:
        for trophy_loc in self.fruits_locations:
            state[-1, trophy_loc[0], trophy_loc[1]] = color

        return state

    def is_cell_empty(self, coordinates):
        return coordinates not in self.grid_map

    def is_within_grid_boundaries(self, point: tuple):
        """
        Checks if a given point is within the grid's boundaries (is it a valid coordinates)
        :param point: (row, col) tuple.
        :return: Boolean
        """
        row, col = point[0], point[1]
        if row < 0 or row >= self.board_size.height:
            return False
        if col < 0 or col >= self.board_size.width:
            return False

        return True

    @staticmethod
    def _build_grid_map(snakes, fruits_locations):
        grid_map = {}
        for i, snake in enumerate(snakes):
            for pos in snake.position:
                assert pos not in grid_map
                grid_map[pos] = i

        for fruit_pos in fruits_locations:
            assert fruit_pos not in grid_map
            grid_map[fruit_pos] = GameState.FRUIT_VALUE

        return grid_map

    def _get_empty_board(self):
        """
        The grid has a layer for each agent + a layer for the trophies.
        the layer at index 0 is for the playing agent (from his perspective), layers 1 to n are for the rest of the
        agents, and the last layer is for the trophies.
        :return:
        """
        grid = np.zeros((self.n_agents + 1, self.board_size.height, self.board_size.width), dtype=np.float32)
        return grid


def get_next_state(game_state: GameState, living_players_actions: dict) -> GameState:
    """

    :param game_state: a GameState object that represents a certain state of the game, which you desire to know the
    next state obtained after performing the given actions upon this given game state.
    :param living_players_actions: a dictionary with an action for each living player.
    {player_index [int] ==> action [GameAction]}
    :return: GameState object. the returned object is a separate copy! This means it has memory implications!
    """
    assert set(game_state.living_agents) == set(living_players_actions.keys())
    assert len(living_players_actions) > 0
    next_state = copy.deepcopy(game_state)
    SnakesBackendSync.perform_env_step(next_state, living_players_actions)
    return next_state


class Player(ABC):
    n_players: int = 0

    @abstractmethod
    def get_action(self, state: GameState) -> GameAction:
        pass

    def __init__(self):
        self.player_index: int = self.n_players
        Player.n_players += 1


PlayersList = List[Player]


class SnakesBackendSync:
    """
    Objects of this class are managing the game. It stores the state of the grid, the snakes etc. It handles the
    actions being made by the agents.

    There are 3 possible actions the agents can perform:
        0 --> Left turn
        1 --> Continue straight
        2 --> Right turn
    """
    FRUIT_VALUE = -1
    SNAKE_BODY = -1
    SNAKE_HEAD = 1

    def __init__(
            self,
            agents: PlayersList,
            grid_size=Grid2DSize(200, 200),
            safe_start_block_size=Grid2DSize(7, 7),
            n_fruits=5,
            game_duration_in_turns=1000,
            random_seed=None
    ):
        self._agents_controllers = agents
        self.n_agents = len(agents)
        self.board_size = grid_size
        self.safe_start_block_size = safe_start_block_size
        self.n_trophies = n_fruits
        self.game_duration_in_turns = game_duration_in_turns
        self._random_seed = random_seed

        self.game_state: GameState = None
        self.awaiting_new_game = []
        self.played_this_turn = []

        self._ensure_sufficient_grid_size()

        self._longest_snake_at_turn_n: WinnerAtTurnList = []

        self.viewer = None

        self.reset_game()

    def reset_game(self):
        """
        Restart the game.
        :return:
        """
        np.random.seed(self._random_seed)
        initial_positions = self._generate_initial_positions()

        for agent_i, agent in enumerate(self._agents_controllers):
            agent.player_index = agent_i

        snakes = [SnakeAgent(agent_i, initial_positions[agent_i]) for agent_i in range(self.n_agents)]

        self.game_state = GameState(turn_number=0,
                                    game_duration_in_turns=self.game_duration_in_turns,
                                    board_size=self.board_size,
                                    current_winner=None,
                                    fruits_locations=[],
                                    snakes=snakes)

        self._fill_fruits(self.game_state)

    @staticmethod
    def perform_env_step(game_state: GameState, living_agents_actions: dict):
        """
        Perform necessary operations for computing the new environment state based on the actions of the agents.
        :return:
        """
        assert set(living_agents_actions.keys()) == set(game_state.living_agents)
        assert not game_state.is_terminal_state

        for agent_i, action in living_agents_actions.items():
            game_state.snakes[agent_i].next_action = action

        game_state.turn_number += 1

        # Update each snake:
        new_heads_pos = []
        for snake in game_state.snakes:
            if not snake.alive:
                continue
            snake.perform_action()
            new_heads_pos.append((snake.index, snake.head))

            # Update the map (partially) - Remove the tail if necessary:
            if snake.head in game_state.grid_map and game_state.grid_map[snake.head] == game_state.FRUIT_VALUE:
                # The snake should grow! do not clip his tail, only replace the fruit by the snake's head (done later):
                # self.grid_map[snake.head] = snake.index
                del game_state.grid_map[snake.head]
                game_state.fruits_locations.remove(snake.head)
            else:
                tail_pos = snake.tail_position
                assert tail_pos in game_state.grid_map
                del game_state.grid_map[tail_pos]
                snake.pop_tail()

        # Update map, handle dead snakes and eaten trophies:
        # Kill snakes that collided with their heads:
        snakes_to_kill = []
        collision_map = {}
        for (i, head) in new_heads_pos:
            if head in collision_map:
                snakes_to_kill.append(i)
                snakes_to_kill.append(collision_map[head])
                continue
            else:
                collision_map[head] = i

            # Kill snakes that ran outside the boundaries:
            if not game_state.is_within_grid_boundaries(head):
                snakes_to_kill.append(i)

            # Kill snakes that collided with other snakes or themselves:
            if head in game_state.grid_map:
                # In this case the snake collided with other snake or itself, kill it:
                # note that we already dealt with the case where the snake eat a fruit!
                snakes_to_kill.append(i)

        for snake_idx in snakes_to_kill:
            SnakesBackendSync._kill_snake(game_state, snake_idx)

        # Update map with living snakes' heads:
        for snake in game_state.snakes:
            if not snake.alive:
                continue
            assert snake.head not in game_state.grid_map
            game_state.grid_map[snake.head] = snake.index

        # Generate new trophies if needed:
        # self._fill_fruits()

        # Determine current winner:
        longest_snake = SnakesBackendSync.determine_longest_at_turn_n(game_state)
        longest_snake_length = game_state.snakes[longest_snake].length if longest_snake is not None else None
        longest_this_turn = WinnerAtTurn(longest_snake, longest_snake_length)
        logging.info(f"Longest snake this turn: {longest_this_turn}")
        if longest_snake is not None:
            if game_state.current_winner is None or game_state.current_winner.length <= longest_snake_length:
                game_state.current_winner = longest_this_turn

    def run_game(self, human_speed=False, render=True):
        if render:
            self.render()
        while self.game_state.turn_number < self.game_duration_in_turns:
            if self._get_num_of_living_snakes() == 0:
                break
            if human_speed:
                time.sleep(0.1)
            agents_actions = {
                agent_index: agent_controller.get_action(self.game_state)
                for agent_index, agent_controller in enumerate(self._agents_controllers)
                if self.game_state.snakes[agent_index].alive
            }
            logging.info("All living players performed actions, performing environment step...")
            self.perform_env_step(self.game_state, agents_actions)
            if render:
                self.render()

            logging.info(f"Current Winner: {self.game_state.current_winner}")

            self.played_this_turn = []

        print(f"Winner: {self.game_state.current_winner}")

    def get_living_agents(self):
        """
        Return a list of indices of the living agents.
        :return: list of integers representing the indices of living agents in the current game.
        """
        return self.game_state.living_agents

    @staticmethod
    def determine_longest_at_turn_n(board_state: GameState):
        """
        Determine which snake is longest in current state. Tie break using 'closest head to a fruit'
        :return:
        """
        longest_snake_idx = None
        for snake_i, snake in enumerate(board_state.snakes):
            if snake.alive:
                if longest_snake_idx is None or snake.length > board_state.snakes[longest_snake_idx].length:
                    longest_snake_idx = snake_i
                    continue
                if snake.length == board_state.snakes[longest_snake_idx].length:
                    # Tie Break
                    snake_manhattan_dists = sorted([cityblock(snake.head, trophy_i)
                                                    for trophy_i in board_state.fruits_locations])
                    longest_manhattan_dists = sorted([cityblock(board_state.snakes[longest_snake_idx].head, trophy_i)
                                                      for trophy_i in board_state.fruits_locations])

                    for d1, d2 in zip(snake_manhattan_dists, longest_manhattan_dists):
                        if d1 < d2:
                            longest_snake_idx = snake_i
                            break
                        elif d1 > d2:
                            break
                        else:
                            # equal distance, tie break with later trophy..
                            pass
        return longest_snake_idx

    def render(self, mode='human'):
        enabled = True
        if enabled and mode == 'human':
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()

            player_index_to_view = 0
            state = self.game_state.get_board(player_index_to_view)
            state[state > 1] = self.SNAKE_BODY
            obs = SnakesBackendSync.convert_observation_to_img(state)

            self.viewer.imshow(obs)

    @staticmethod
    def convert_observation_to_img(observation):
        if not isinstance(observation, np.ndarray):
            print(observation)
        assert isinstance(observation, np.ndarray)
        image_shape = observation.shape[1:] + (3,)
        white_img = np.ones(image_shape) * 255
        player_channel, opponents_channel, trophies_channel = 2, 0, 1

        rgb_min, rgb_max = 0, 255
        scale_factor = rgb_max * 0.6

        transform_values_to_rgb = lambda x: (x < 0) * scale_factor
        transform_values_to_rgb_mask = lambda x: (x < 0) * rgb_max + (x > 0) * rgb_max * 0.6
        observation_transformed = np.zeros(observation.shape)
        observation_mask = np.zeros(observation.shape)
        for i in range(len(observation)):
            observation_transformed[i] = transform_values_to_rgb(observation[i])
            observation_mask[i] = transform_values_to_rgb_mask(observation[i])

        white_img[:, :, player_channel] -= observation_transformed[0]
        white_img[:, :, opponents_channel] -= observation_mask[0]
        white_img[:, :, trophies_channel] -= observation_mask[0]

        for i in range(1, 2 + len(observation) - 3):
            white_img[:, :, opponents_channel] -= observation_transformed[i]
            white_img[:, :, player_channel] -= observation_mask[i]
            white_img[:, :, trophies_channel] -= observation_mask[i]
        white_img[:, :, trophies_channel] -= observation_transformed[-1]
        white_img[:, :, player_channel] -= observation_mask[-1]
        white_img[:, :, opponents_channel] -= observation_mask[-1]

        img = white_img.astype(np.uint8)
        return np.array(Image.fromarray(img).resize((observation.shape[2] * 8, observation.shape[1] * 8)))

    def _ensure_sufficient_grid_size(self):
        max_num_of_agents = int(self.board_size.width / self.safe_start_block_size.width) * \
                            int(self.board_size.height / self.safe_start_block_size.height)
        if self.n_agents > max_num_of_agents:
            raise GridTooSmallException()

    def _generate_initial_positions(self):
        #	First, sample initial blocks:
        max_num_of_agents = int(self.board_size.width / self.safe_start_block_size.width) * int(
            self.board_size.height / self.safe_start_block_size.height)
        initial_blocks = np.random.choice(max_num_of_agents, size=self.n_agents, replace=False)

        #	generate for each block it's position:
        initial_cols = initial_blocks % (self.board_size.width // self.safe_start_block_size.width)
        initial_rows = initial_blocks // (self.board_size.width // self.safe_start_block_size.width)

        w = initial_cols * self.safe_start_block_size.width + self.safe_start_block_size.width // 2
        h = initial_rows * self.safe_start_block_size.height + self.safe_start_block_size.height // 2

        return [(h_i, w_i) for w_i, h_i in zip(w, h)]

    def _fill_fruits(self, board_state: GameState):
        while len(board_state.fruits_locations) < self.n_trophies:
            row = np.random.randint(0, board_state.board_size.height)
            col = np.random.randint(0, board_state.board_size.width)

            while not board_state.is_cell_empty((row, col)):
                row = np.random.randint(0, board_state.board_size.height)
                col = np.random.randint(0, board_state.board_size.width)

            trophy_pos = (row, col)
            board_state.fruits_locations.append(trophy_pos)
            board_state.grid_map[trophy_pos] = board_state.FRUIT_VALUE

    @staticmethod
    def _kill_snake(board_state, index, replace_body_with_trophies=True):
        """
        Do all procedures needed to set the snake's state to dead
        :param index:
        :return:
        """
        # Kill it:
        board_state.snakes[index].kill()

        # Update the map:
        for pos in board_state.snakes[index].position:
            if pos in board_state.grid_map and board_state.grid_map[pos] == index:
                del board_state.grid_map[pos]
                if replace_body_with_trophies:
                    board_state.fruits_locations.append(pos)
                    board_state.grid_map[pos] = board_state.FRUIT_VALUE

    def _get_num_of_living_snakes(self):
        return len(self.game_state.living_agents)
