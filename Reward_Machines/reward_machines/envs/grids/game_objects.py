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
    

