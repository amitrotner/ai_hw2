from environment import Grid2DSize, SnakesBackendSync
from agents import RandomPlayer, KeyboardPlayer, GreedyAgent, BetterGreedyAgent
from submission import MinimaxAgent, AlphaBetaAgent, TournamentAgent
from optparse import OptionParser
import sys


def start_demo_game(n_agents: int, game_duration: int, board_width: int, board_height: int,
                    n_fruits: int, use_keyboard_listener: bool):
    players = [KeyboardPlayer(use_keyboard_listener=use_keyboard_listener)] + [GreedyAgent() for _ in
                                                                               range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=not use_keyboard_listener)


def start_game_with_players(players, game_duration: int, board_width: int, board_height: int, n_fruits: int,
                            fast_run: bool = False, graphics_off: bool = False):
    if len(players) < 1:
        print("The number of agents must be at least 1.")

    env = SnakesBackendSync(players,
                            grid_size=Grid2DSize(board_width, board_height),
                            n_fruits=n_fruits,
                            game_duration_in_turns=game_duration)
    env.run_game(human_speed=not fast_run, render=not graphics_off)


def start_part_c(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [BetterGreedyAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_part_d(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [MinimaxAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_part_e(n_agents: int, game_duration: int, board_width: int, board_height: int,
                 n_fruits: int, fast_run: bool, graphics_off: bool):
    players = [AlphaBetaAgent()] + [GreedyAgent() for _ in range(n_agents - 1)]
    start_game_with_players(players,
                            game_duration,
                            board_width,
                            board_height,
                            n_fruits,
                            fast_run=fast_run,
                            graphics_off=graphics_off)


def start_custom_game(p1: str, p2: str, game_duration: int, board_width: int, board_height: int,
                      n_fruits: int, fast_run: bool, graphics_off: bool, use_keyboard_listener: bool):
    def get_player(p: str):
        if p == 'KeyboardPlayer':
            return KeyboardPlayer(use_keyboard_listener=use_keyboard_listener)
        elif p == 'GreedyAgent':
            return GreedyAgent()
        elif p == 'BetterGreedyAgent':
            return BetterGreedyAgent()
        elif p == 'MinimaxAgent':
            return MinimaxAgent()
        elif p == 'AlphaBetaAgent':
            return AlphaBetaAgent()
        elif p == 'TournamentAgent':
            return TournamentAgent()

    players = [get_player(p1), get_player(p2)]

    fast_run = True if use_keyboard_listener == False else fast_run
    for i in range(10):
        start_game_with_players(players,
                                game_duration,
                                board_width,
                                board_height,
                                n_fruits,
                                fast_run=fast_run,
                                graphics_off=graphics_off)

def get_user_command(argv):
    usage_str = """
    USAGE: python main.py <options>
    """
    parser = OptionParser(usage_str)

    parser.add_option('-d', '--demo', dest='demo_run',
                      help="Run a demo game, with interactive keyboard",
                      default=True)
    parser.add_option('-f', '--fast', dest='fast_run', default=False, action='store_true', help='Run Fast')
    parser.add_option('-g', '--graphics_off', dest='graphics_off', default=False, action='store_true')
    parser.add_option('--n_agents', dest='n_agents', type=int,
                      help="number of snakes in the game", default=2)
    parser.add_option('-N', '--game_duration', dest='game_duration', type=int,
                      help="Number of turns in a game", default=500)
    parser.add_option('-W', '--board-width', dest='board_width', type=int,
                      help="width of the board", default=50)
    parser.add_option('-H', '--board-height', dest='board_height', type=int,
                      help="height of the board", default=50)
    parser.add_option('--n_fruits', dest='n_fruits', type=int,
                      help="number of fruits on the board", default=51)
    parser.add_option('--kb_listener_off', dest='use_keyboard_listener', default=True, action='store_false',
                      help="Whether or not to use keyboard listener for logging KB events. If you use Mac OS you MUST use False!")
    parser.add_option('--part_c', dest='part_c', default=False, action='store_true',
                      help="whether to run part C.")
    parser.add_option('--part_d', dest='part_d', default=False, action='store_true',
                      help="whether to run part D.")
    parser.add_option('--part_e', dest='part_e', default=False, action='store_true',
                      help="whether to run part E.")
    parser.add_option('--custom_game', dest='custom_game', default=False, action='store_true',
                      help="whether to run a custom game with user defined players.")
    parser.add_option('--p1', dest='player1', default='KeyboardPlayer',
                      help="The class name of player 1 in the custom game")
    parser.add_option('--p2', dest='player2', default='GreedyPlayer',
                      help="The class name of player 2 in the custom game")
    options, other = parser.parse_args(argv)
    if len(other):
        raise Exception(f'wrong usage: {other}')

    if options.part_c:
        start_part_c(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.part_d:
        start_part_d(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.part_e:
        start_part_e(options.n_agents,
                     options.game_duration,
                     options.board_width,
                     options.board_height,
                     options.n_fruits,
                     options.fast_run,
                     options.graphics_off)
    elif options.custom_game:
        start_custom_game(options.player1,
                          options.player2,
                          options.game_duration,
                          options.board_width,
                          options.board_height,
                          options.n_fruits,
                          options.fast_run,
                          options.graphics_off,
                          options.use_keyboard_listener
                          )
    elif options.demo_run:
        start_demo_game(options.n_agents,
                        options.game_duration,
                        options.board_width,
                        options.board_height,
                        options.n_fruits,
                        options.use_keyboard_listener)


if __name__ == '__main__':
    get_user_command(sys.argv[1:])
