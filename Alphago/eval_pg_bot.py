import argparse
import h5py
from collections import namedtuple

from dlgo import agent
from dlgo import scoring
from dlgo.goboard_fast import GameState, Player


BOARD_SIZE = 19


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


