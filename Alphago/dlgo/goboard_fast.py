import copy
from dlgo.gotypes import Player, Point
from dlgo.scoring import compute_game_result
from dlgo import zobrist
from dlgo.utils import MoveAge

__all__ = [
    'Board',
    'GameState',
    'Move',
]

neighbor_tables = {}
corner_tables = {}


def init_neighbor_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_neighbors = p.neighbors()
            true_neighbors = [
                n for n in full_neighbors
                if 1 <= n.row <= rows and 1 <= n.col <= cols
            ]
            new_table[p] = true_neighbors
    neighbor_tables[dim] = new_table


def init_corner_table(dim):
    rows, cols = dim
    new_table = {}
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            p = Point(row=r, col=c)
            full_corners = [
                Point(row=p.row - 1, col=p.col - 1),
                Point(row=p.row - 1, col=p.col + 1),
                Point(row=p.row + 1, col=p.col - 1),
                Point(row=p.row + 1, col=p.col + 1),
            ]
            true_corners = [
                n for n in full_corners
                if 1 <= n.row <= rows and 1 <= n.col <= cols
            ]
            new_table[p] = true_corners
    corner_tables[dim] = new_table


class IllegalMoveError(Exception):
    pass


class GoString():
    def __init__(
            self,
            color,
            stones,
            liberties):
        
        self.color = color
        self.stones = stones
        self.liberties = liberties
    
    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])
        return GoString(self.color, self.stones, new_liberties)
    
    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])
        return GoString(self.colir, self.stones, new_liberties)
    
    def merged_with(self, string):
        assert string.color == self.color 
        combined_stones = self.stones | string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | string.liberties) - combined_stones
        )
    
    @property
    def num_liberties(self):
        return len(self.liberties)
    
    def __eq__(self, other):
        return isinstance(other, GoString) and \
            self.color == other.color and \
            self.stones == other.stones and \
            self.liberties == other.liberties
    
    def __deepcopy__(self, memodict={}):
        return GoString(
            self.color,
            self.stones,
            copy.deepcopy(self.liberties)
        )


