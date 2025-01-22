import numpy as np

from dlgo.encoders.base import Encoder
from dlgo.encoders.utils import is_ladder_example, is_ladder_capture
from dlgo.gotypes import Point, Player
from dlgo.goboard_fast import Move
from dlgo.agent.helpers import is_point_an_eye

FEATRE_OFFSETS = {
    'stone_color': 0,
    'ones': 3,
    'zeros': 4,
    'sensibleness': 5,
    'turns_since': 6,
    'liberties': 14,
    'liberties_after': 22,
    'capture_size': 30,
    'self_atari_size': 38,
    'ladder_capture': 46,
    'ladder_escape': 47,
    'current_player_color': 48,
}


def offset(feature):
    return FEATRE_OFFSETS[feature]


