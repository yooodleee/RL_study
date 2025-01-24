import argparse
import h5py
from collections import namedtuple

from dlgo import agent
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player


class GameRecord(namedtuple('GameRecord', 'Winner')):
    pass


def simulate_game(black_player, white_player):
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        game = game.apply_move(next_move)
    game_result = scoring.compute_game_result(game)

    print(game_result)
    
    return GameRecord(
        Winner=game_result.winner
    )


