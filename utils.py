from environment import SnakesBackendSync, Grid2DSize
from agents import GreedyAgent, StaticAgent, RandomPlayer
import numpy as np


def get_fitness(moves_sequence: tuple) -> float:
    n_agents = 20
    static_agent = StaticAgent(moves_sequence)
    opponents = [RandomPlayer() for _ in range(n_agents - 1)]
    players = [static_agent] + opponents

    board_width = 40
    board_height = 40
    n_fruits = 50
    game_duration = len(moves_sequence)

    env = SnakesBackendSync(players,
                            grid_size=Grid2DSize(board_width, board_height),
                            n_fruits=n_fruits,
                            game_duration_in_turns=game_duration, random_seed=42)
    env.run_game(human_speed=False, render=False)
    np.random.seed()
    return env.game_state.snakes[0].length + env.game_state.snakes[0].alive
