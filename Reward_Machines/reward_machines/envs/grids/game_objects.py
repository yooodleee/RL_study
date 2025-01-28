from enum import Enum
import random


class Entity:

    def __init__(self, i, j):   # row and column
        self.i = i
        self.j = j
    
    def change_position(self, i, j):
        self.i = i
        self.j = j
    
    def idem_position(self, i, j):
        return self.i == i and self.j == j
    
    def interact(self, agent):
        return True
    

class Agent(Entity):

    def __init__(self, i, j, actions):
        super().__init__(i, j)
        self.actions = actions
        self.initial_position = (i, j)
    
    def reset(self):
        self.change_position(*self.initial_position)
    
    def get_actions(self):
        return self.actions
    
    def __str__(self):
        return "A"
    

class Obstacle(Entity):

    def __init__(self, i, j):
        super().__init__(i, j)
    
    