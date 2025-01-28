import gym
import random
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.grids.craft_world import CraftWorld
from envs.grids.office_world import OfficeWorld
from envs.grids.value_iteration import value_iteration


class GridEnv(gym.Env):

    def __init__(self, env):
        self.env = env
        N, M = self.env.map_heigth, self.env.map_width
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Box(
            low=0, high=max([N, M]), shape=(2,), dtype=np.uint8
        )
    
    def get_events(self):
        return self.env.get_true_propositions()
    
    def step(self, action):
        self.env.execute_action(action)
        obs = self.env.get_features()
        reward = 0  # all the reward comes from the RM
        done = False
        info = {}
        
        return obs, reward, done, info
    
    def reset(self):
        self.env.reset()
        return self.env.get_features()
    
    def show(self):
        self.env.show()
    
    def get_model(self):
        return self.env.get_model()


