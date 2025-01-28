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
    
    def get_features(self):
        """
        returns the features of the current stat (i.e., the location of the agent)
        """
        x, y = self.agent
        return np.array([x, y])
    
    def show(self):
        for y in range(8, -1, -1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.up) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                
                print()
            for x in range(12):
                if (x, y, Actions.left) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 0:
                    print(" ", end="")
                if (x, y) == self.agent:
                    print("A", end="")
                elif (x, y) in self.objects:
                    print(self.objects[(x, y)], end="")
                else:
                    print(" ", end="")
                if (x, y, Actions.right) in self.forbidden_transitions:
                    print("|", end="")
                elif x % 3 == 2:
                    print(" ", end="")
            print()
            if y % 3 == 0:
                for x in range(12):
                    if x % 3 == 0:
                        print("_", end="")
                        if 0 < x < 11:
                            print("_", end="")
                    if (x, y, Actions.down) in self.forbidden_transitions:
                        print("_", end="")
                    else:
                        print(" ", end="")
                print()
    
    def get_model(self):
        """
        returns a model of the environment.
        Compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per step
            of each task to 1.
        """
        S = [(x, y) for x in range(12) for y in range(9)]   # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}  # Transitions (s, a) -> s' (they are deterministic)
        for s in S:
            x, y = S
            for a in A:
                T[(s, a)] = self._get_new_position(x, y, a)
        
        return S, A, L, T   # SALT xD
    
    