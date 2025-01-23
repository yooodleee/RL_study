import random

from dlgo.agent import Agent
from dlgo.scoring import GameResult


__all__ = [
    'DepthprunedAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999


def reserve_game_result(game_result):
    if game_result == GameResult.loss:
        return game_result.win
    if game_result == GameResult.win:
        return game_result.loss
    return GameResult.draw


