from keras import layers    # Conv2D, BatchNormalization, Dense, Flatten, Input
from keras import models    # Model
from dlgo.goboard_fast import GameState, Player
from dlgo import scoring 
from dlgo import zero


def simulate_game(
        board_size,
        black_agent, black_collector,
        white_agent, white_collector):
    
    print('Starting the game!')
    game = GameState.new_game(board_size)
    agents = {
        Player.black: black_agent,
        Player.white: white_agent,
    }

    black_collector.begin_episode()
    white_collector.begin_episode()
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    
    game_result = scoring.compute_game_result(game)
    print(game_result)
    if game_result.winner == Player.black:
        black_collector.complete_episode(1)
        white_collector.complete_episode(-1)
    else:
        black_collector.complete_episode(-1)
        white_collector.complete_episode(1)
    

