import argparse
import h5py
from collections import namedtuple

from dlgo import rl
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player


BOARD_SIZE = 19


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


def simulate_game(black_player, white_player):
    moves = []
    game = GameState.new_game(BOARD_SIZE)
    agents = {
        Player.black: black_player,
        Player.white: white_player,
    }
    while not game.is_over():
        next_move = agents[game.next_player].select_move(game)
        moves.append(next_move)
        game = game.apply_move(next_move)
    game_result = scoring.compute_game_result(game)
    print(game_result)
    return GameRecord(
        moves=moves,
        winner=game_result.winner,
    )


