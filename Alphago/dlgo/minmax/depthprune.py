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


def best_result(
        game_state,
        max_depth,
        eval_fn):
    
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE
    
    if max_depth == 0:
        return eval_fn(game_state)
    
    best_so_far = MIN_SCORE
    for candidate_move in game_state.legal_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = best_result(
            next_state,
            max_depth,
            -1,
            eval_fn,
        )
        our_result = -1 * opponent_best_result
        if our_result > best_so_far:
            best_so_far = our_result
    
    return best_so_far


