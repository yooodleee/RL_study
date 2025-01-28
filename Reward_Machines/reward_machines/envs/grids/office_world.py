from envs.grids.game_objects import Actions
import random
import math
import os
import numpy as np


class OfficeWorld:

    def __init__(self):
        self._load_map()
        self.map_height, self.map_width = 12, 9
    
    