import random

from dlgo.agent import Agent
from dlgo.gotypes import Player


__all__ = [
    'AlphaBetaAgent',
]

MAX_SCORE = 999999
MIN_SCORE = -999999


def alpha_beta_result(
        game_state,
        max_depth,
        best_black,
        best_white,
        eval_fn):
    
    if game_state.is_over():
        if game_state.winner() == game_state.next_player:
            return MAX_SCORE
        else:
            return MIN_SCORE
    
    if max_depth == 0:
        return eval_fn(game_state)
    
    best_so_far = MIN_SCORE
    for candidate_move in game_state.leval_moves():
        next_state = game_state.apply_move(candidate_move)
        opponent_best_result = alpha_beta_result(
            next_state,
            max_depth,
            -1,
            best_black,
            best_white,
            eval_fn,
        )
        our_result = -1 * opponent_best_result

        if our_result > best_so_far:
            best_so_far = our_result
        
        if game_state.next_player == Player.white:
            if best_so_far > best_white:
                best_white = best_so_far
            outcome_for_black = -1 * best_so_far
            if outcome_for_black < best_black:
                return best_so_far
        elif game_state.next_player == Player.black:
            if best_so_far > best_black:
                best_black = best_so_far
            outcome_for_white = -1 * best_so_far
            if outcome_for_white < best_white:
                return best_so_far
    
    return best_so_far


