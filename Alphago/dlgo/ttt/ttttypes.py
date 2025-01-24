import enum
from collections import namedtuple


__all__ = [
    'Player',
    'Point',
]


class Player(enum.Enum):
    x = 1
    o = 2

    @property
    def other(self):
        return Player.x if self == Player.o else Player.o


