from envs.grids.game_objects import *
import random
import math
import os


class CraftWorld:

    def __init__(self, file_map):
        self.file_map = file_map
        self._load_mal(file_map)
        self.env_game_over = False
    
    def reset(self):
        self.agent.reset()
    
    def execute_action(self, a):
        """
        execute 'action' in the game.
        """
        agent = self.agent
        ni, nj = agent.i, agent.j

        # Getting new position after executing action
        ni, nj = self._get_next_position(ni, nj, a)

        # Interacting with the objects that is in the next position
        # (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni, nj)
    
    def _get_next_position(self, ni, nj, a):
        """
        Returns the position where the agent would be if we execute action.
        """
        action = Actions(a)

        # OBS: Invalid actions behave as NO-OP
        if action == Actions.up     : ni -= 1
        if action == Actions.down   : ni += 1
        if action == Actions.left   : nj -= 1
        if action == Actions.right  : nj += 1

        return ni, nj
    