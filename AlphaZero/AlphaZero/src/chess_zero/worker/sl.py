import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from logging import getLogger
from threading import Thread
from time import time
from typing import Union

import chess.pgn

from chess_zero.agent.player_chess import ChessPlayer
from config import Config
from chess_zero.env.chess_env import ChessEnv, Winner
from chess_zero.lib.data_helper import write_game_data_to_file, find_pgn_files

logger = getLogger(__name__)

TAG_REGEX = re.compile(r"^\[([A-Za-z0-9_]+)\s+\"(.*)\"\]\s*$")


def start(config: Config):
    return SupervisedLearningWorker(config).start()



class SupervisedLearningWorker:
    def __init__(self):
        self.config = Config
        self.buffer = []

    def start(self):
        self.buffer = []
        # noinspection PyAttributeOutsideInit
        self.idx = 0
        start_time = time()
        with ProcessPoolExecutor(max_workers=7) as executor:
            games = self.get_games_from_all_files()
            # poisned reference (memleak)
            for res in as_completed([executor.submit(get_buffer, \
                                                     self.config, game)
                                                       for game in games]):    
                self.idx += 1
                env, data = res.result()
                self.save_data(data)
                end_time = time()
                logger.debug(f"game {self.idx:4} 
                             time={(end_time - start_time):.3f}s "
                             f"halfmoves={env.num_halfmoves:3} 
                             {env.winner:12}"
                             f"{' by resign ' if env.resigned 
                                else '            '}"
                             f"{env.observation.split(' ')[0]}")
                start_time = end_time
            
        if len(self.buffer) > 0:
            self.flush_buffer()
    
    def get_games_from_all_files(self):
        files = find_pgn_files(self.config.resource.play_data_dir)
        print(files)
        games = []
        for filename in files:
            games.extend(get_games_from_file(filename))
        print("done reading")
        return games
    
    def save_data(self, data):
        self.buffer += data
        if self.idx % self.config.play_data.sl_nb_game_in_file == 0:
            self.flush_buffer()
    
    def flush_buffer(self):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        path = os.path.join(rc.play_data_dir, 
                            rc.play_data_filename_tmpl % game_id)
        logger.info(f"save play data to {path}")
        thread = Thread(target=write_game_data_to_file(), 
                        args=(path, self.buffer))
        thread.start()
        self.buffer = []


def get_games_from_file(filename):
    pgn = open(filename, errors='ignore')
    offsets = list(chess.pgn.scan_offsets(pgn))
    n = len(offsets)
    print(f"found {n} games")
    games = []
    for offset in offsets:
        pgn.seek(offset)
        games.append(chess.pgn.read_game(pgn))
    return games


def clip_elo_policy(config, elo):
    return min(1, max(0, elo - config.play_data.min_elo_policy) \
               / config.play_data.max_elo_policy)
# 0 until min_elo, 1 after max_elo, linear in between


def get_buffer(config, game)-> Union[ChessEnv, list]:
    env = ChessEnv().reset()
    white = ChessPlayer(config, dummy=True)
    black = ChessPlayer(config, dummy=True)
    result = game.headers["Result"]
    white_elo, black_elo = int(game.headers["WhiteElo"]), \
                            int(game.headers["BlackElo"])
    white_weight = clip_elo_policy(config, white_elo)
    black_weight = clip_elo_policy(config, black_elo)

    actions = []
    while not game.is_end():
        game = game.variation(0)
        actions.append(game.move.uci())
    k = 0
    while not env.done and k < len(actions):
        if env.white_to_move:
            action = white.sl_action(env.observation, 
                                     actions[k], 
                                     weight=white_weight)  # ignore=True
        else:
            action = black.sl_action(env.observation, 
                                     actions[k], 
                                     weight=black_weight)  # ignore=True
        
        env.step(action, False)
        k += 1
    
    if not env.board.is_game_over() and result != '1/2-1/2':
        env.resigned = True
    if result == '1-0':
        env.winner = Winner.white
        black_win = -1
    elif result == '0-1':
        env.winner = Winner.black
        black_win = 1
    else:
        env.winner = Winner.draw
        black_win = 0
    
    black.finish_game(black_win)
    white.finish_game(-black_win)

    data = []
    for i in range(len(white.moves)):
        data.append(white.moves[i])
        if i < len(black.moves):
            data.append(black.moves[i])
    
    return env, data