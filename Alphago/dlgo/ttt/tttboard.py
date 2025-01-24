import copy

from dlgo.ttt.ttttypes import Player, Point


__all__ = [
    'Board',
    'GameState',
    'Move',
]


class IllegalMoveError(Exception):
    pass


BOARD_SIZE = 3
ROWS = tuple(range(1, BOARD_SIZE + 1))
COLS = tuple(range(1, BOARD_SIZE + 1))
DIAG_1 = (Point(1, 1), Point(2, 2), Point(3, 3))
DIAG_2 = (Point(1, 3), Point(2, 2), Point(3, 1))


