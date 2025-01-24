from collections import namedtuple

from dlgo import rl
from dlgo import scoring
from dlgo import goboard_fast as goboard
from dlgo.gotypes import Player


class GameRecord(namedtuple('GameRecord', 'moves winner')):
    pass


