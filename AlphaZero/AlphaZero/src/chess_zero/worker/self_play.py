import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time

from chess_zero.agent.model_chess import ChessModel
from chess_zero.agent.player_chess import ChessPlayer
from config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper import get_game_data_filenames, write_game_data_to_file, pretty_print
from chess_zero.lib.model_helper import load_best_model_weight, save_as_best_model, reload_best_model_weight_if_changed

logger = getLogger(__name__)

def start(config: Config):
    return SelfPlayWorker(config).start()


# noinspection PyAttributeOutsideInit
class SelfPlayWorker:
    def __init__(self, config: Config):
        self.config = config
        self.current_model = self.load_model()
        self.m = Manager()
        self.cur_pipes = self.m.list(
            [self.current_model.get_pipes(self.config.play.search_threads) for _ in range(self.config.play.max_processes)])
        