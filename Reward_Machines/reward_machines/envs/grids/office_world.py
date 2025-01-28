from envs.grids.game_objects import Actions
import random
import math
import os
import numpy as np


class OfficeWorld:

    def __init__(self):
        self._load_map()
        self.map_height, self.map_width = 12, 9
    
    def reset(self):
        self.agent = (2, 1)
    
    def execute_action(self, a):
        """
        execute 'action' in the game.
        """
        x, y = self.agent
        self.agent = self._get_new_position(x, y, a)

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x, y, action) not in self.forbidden_transitions:
            if action == Actions.up     : y += 1
            if action == Actions.down   : y -= 1
            if action == Actions.left   : x -= 1
            if action == Actions.right  : x += 1
    
        return x, y
    
    def get_true_propositions(self):
        """
        returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        return ret
    
    