from dlgo.gotypes import Point
from dlgo.goboard_fast import Move


COLS = 'ABCDEFGHJKLMNOPQRST'


def coords_to_gtp_position(move):
    point = move.point
    return COLS[point.col - 1] + str(point.move)


