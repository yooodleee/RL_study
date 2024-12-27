from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from logging import getLogger
from threading import Lock
from typing import Union, Tuple

import chess
import numpy as np

from chess_zero.config import Config
from chess_zero.env.chess_env import ChessEnv, Winner

# from chess_zero.play_game.uci import info

logger = getLogger(__name__)

# these are from AGZ nature paper
class VisitStats:
    def __init__(self):
        self.a = defaultdict(ActionStats)
        self.sum_n = 0


class ActionStats:
    def __init__(self):
        self.n = 0
        self.w = 0
        self.q = 0


class ChessPlayer:
    # dot = False
    def __init__(
        self,
        config: Config,
        pipes = None,
        play_config = None,
        dummy = False):
        self.moves = []

        self.config = config
        self.play_config = play_config or self.config.play
        self.labels_n = config.n_labels
        self.labels = config.labels
        self.move_lookup = {
            chess.Move.from_uci(move): i for move, i in zip(self.labels, range(self.labels_n))}
        if dummy:
            return
        
        self.pipe_pool = pipes
        self.node_lock = defaultdict(Lock)
    
    def reset(self):
        self.tree = defaultdict(VisitStats)
    
    def deboog(self, env):
        print(env.testeval())

        state = state_key(env)
        my_visit_stats = self.tree[state]
        stats = []
        for action, a_s in my_visit_stats.a.items():
            moi = self.move_lookup[action]
            stats.append(np.asarray([a_s.n, a_s.w, a_s.q, a_s.p, moi]))
        stats = np.asarray(stats)
        a = stats[stats[:,0].argsort()[::-1]]

        for s in a:
            print(f'{self.labels[int(s[4])]:5}: '
                  f'n: {s[0]:3.0f} '
                  f'w: {s[1]:7.3f} '
                  f'q: {s[2]:7.3f} '
                  f'p: {s[3]:7.5f} ')
    
    def action(self, env, can_stop = True)-> str:
        self.reset()

        # for tl in range(self.play_config.thinking_loop):
        root_value, naked_value = self.search_moves(env)
        policy = self.calc_policy(env)
        my_action = int(
            np.random.choice(range(self.labels_n),
                             p = self.apply_temperature(policy, env.num_halfmoves)))
        # print(naked_value)
        # self.deboog(env)
        if can_stop and self.play_config.resign_threshold is not None and \
                                        root_value <= self.play_config.resign_threshold \
                                        and env.num_halfmoves > self.play_config.min_resign_turn:
            # noinspection PyTypeChecker
            return None
        else:
            self.moves.append([env.observation, list(policy)])
            return self.config.labels[my_action]
    
    def search_moves(self, env)-> Union[float, float]:
        # if ChessPlayer.dot == False:
        #       import stacktracer:
        #       stacktracer.trace_start("trace.html")
        #       ChessPlayer.dot = True

        futures = []
        with ThreadPoolExecutor(max_workers=self.play_config.search_threads) as executor:
            for _ in range(self.play_config.simulation_num_per_move):
                futures.append(executor.submit(self.search_moves, env=env.copy(), is_root_node=True))
        
        vals = [f.result() for f in futures]
        # vals=[self.search_my_move(env.copy(), True) for _ in range(self.play_config.simulation_num_per_move)]

        return np.max(vals), vals[0]    # vals[0] is kind of racy

    def search_my_move(self, env: ChessEnv, is_root_node=False)-> float:
        """
        Q, V is value for this Player(always white).
        P is value for the player of next_player (black or white)

        Return:
            leaf value
        """
        if env.done:
            if env.winner == Winner.draw:
                return 0
            # assert env.whiteon != env.white_to_move
            # side to move can't be winner!
        
        