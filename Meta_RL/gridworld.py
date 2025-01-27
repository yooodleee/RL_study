import numpy as np
import random
import itertools
import scipy.ndimage
import scipy.misc
import matplotlib.pyplot as plt


class gameOb():
    def __init__(
            self,
            coordinates,
            size,
            color,
            reward,
            name):
        
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.color = color
        self.reward = reward
        self.name = name


class gameEnv():
    def __init__(
            self,
            partial,
            size,
            goal_color):
        
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        self.bg = np.zeros([size, size])
        a, a_big = self.reset(goal_color)
        plt.imshow(a_big, interpolation="nearest")
    
    