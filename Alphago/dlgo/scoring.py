from __future__ import absolute_import
from collections import namedtuple

from dlgo.gotypes import Player, Point


class Territory(object):
    def __init__(self, territory_map):
        self.num_black_territory = 0
        self.num_white_territory = 0
        self.num_black_stones = 0
        self.num_white_stones = 0
        self.num_dame = 0
        self.dame_points = []
        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == 'territory_b':
                self.num_black_territory += 1
            elif status == 'territory_w':
                self.num_white_territory += 1
            elif status == 'dame':
                self.num_dame += 1
                self.dame_points.append(point)


