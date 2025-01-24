from dlgo import minmax
from dlgo import ttt

from six.moves import input


COL_NAMES = 'ABC'


def print_board(board):
    print('   A   B   C')
    for row in (1, 2, 3):
        pieces = []
        for col in (1, 2, 3):
            piece = board.get(ttt.Point(row, col))
            if piece == ttt.Player.x:
                pieces.append('X')
            elif piece == ttt.Player.o:
                pieces.append('O')
            else:
                pieces.append(' ')
        print('%d   %s' % (row, ' | '.join(pieces)))


def point_from_coords(text):
    col_name = text[0]
    row = int(text[1])
    return ttt.Point(
        row, COL_NAMES.index(col_name) + 1
    )


