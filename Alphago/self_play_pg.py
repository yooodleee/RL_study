import argparse
import h5py
from collections import namedtuple

from dlgo import agent
from dlgo import scoring
from dlgo import rl
from dlgo.goboard_fast import GameState, Player


class GameRecord(namedtuple('GameRecord', 'Winner')):
    pass


