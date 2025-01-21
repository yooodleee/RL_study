import numpy as np

from dlgo.agent.base import Agent
from dlgo.goboard_fast import Move
from dlgo import kerasutil
import operator


class AlphaGoNode:
    def __init__(
            self,
            parent=None,
            probability=1.0):
        
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_value = 0
        self.prior_value = probability
        self.u_value = probability
    
    def select_child(self):
        return max(
            self.children.items(),
            key=lambda child: child[1].q_values + child[1].u_value,
        )
    
    def expand_children(
            self,
            moves,
            probabilities):
        
        for move, prob in zip(moves, probabilities):
            if move not in self.children:
                self.children[move] = AlphaGoNode(probability=prob)
    
    def update_values(self, leaf_value):
        if self.parent is not None:
            self.parent.update_values(leaf_value)
        
        self.visit_count += 1
        self.q_value += leaf_value / self.visit_count

        if self.parent is not None:
            c_u = 5
            self.u_value = c_u * np.sqrt(self.parent.visit_count) \
                * self.prior_value / (1 + self.visit_count)


