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


class Board:
    def __init__(self):
        self._grid = {}
    
    def place(self, player, point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None
        self._grid[point] = player
    
    @staticmethod
    def is_on_grid(point):
        return 1 <= point.row <= BOARD_SIZE and \
            1 <= point.col <= BOARD_SIZE
    
    def get(self, point):
        return self._grid.get(point)


class Move:
    def __init__(self, point):
        self.point = point
    

class GameState:
    def __init__(
            self,
            board,
            next_player,
            move):
        
        self.board = board
        self.next_player = next_player
        self.last_move = move
    
    def _has_3_in_a_row(self, player):
        for col in COLS:
            if all(
                self.board.get(Point(row, col)) == player for row in ROWS
            ):
                return True
        
        for row in ROWS:
            if all(
                self.board.get(Point(row, col)) == player for col in COLS
            ):
                return True
        
        if self.board.get(Point(1, 1)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 3)) == player:
            return True
        
        if self.board.get(Point(1, 3)) == player and \
                self.board.get(Point(2, 2)) == player and \
                self.board.get(Point(3, 1)) == player:
            return True
    
    def apply_move(self, move):
        next_board = copy.deepcopy(self.board)
        next_board.place(self.next_player, move.point)
        return GameState(
            next_board,
            self.next_player.other,
            move,
        )
    
    @classmethod
    def new_game(cls):
        board = Board()
        return GameState(board, Player.x, None)
    
    def is_valid_move(self, move):
        return (
            self.board.get(move.point) is None and
            not self.is_ober()
        )
    
    def legal_moves(self):
        moves = []
        for row in ROWS:
            for col in COLS:
                move = Move(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        return moves
    
    def is_over(self):
        if self._has_3_in_a_row(Player.x):
            return True
        if self._has_3_in_a_row(Player.o):
            return True
        if all(
            self.board.get(Point(row, col)) is not None
            for row in ROWS
            for col in COLS
        ):
            return True
        return False
    
    def winner(self):
        if self._has_3_in_a_row(Player.x):
            return Player.x
        if self._has_3_in_a_row(Player.o):
            return Player.o
        return None