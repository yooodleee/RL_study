from envs.grids.game_objects import *
import random
import math
import os


class CraftWorld:

    def __init__(self, file_map):
        self.file_map = file_map
        self._load_mal(file_map)
        self.env_game_over = False
    
    