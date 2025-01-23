import enum
import random

from dlgo.agent import Agent


__all__ = [
    'MinmaxAgent',
]


class GameResult(enum.Enum):
    loss = 1
    draw = 2
    win = 3


