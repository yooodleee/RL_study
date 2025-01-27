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
    
    def getFeatures(self):
        return np.array([self.objects[0].x, self.objects[0].y]) / float(self.sizeX)
    
    def reset(self, goal_color):
        self.objects = []
        self.goal_color = goal_color
        self.other_color = [1 - a for a in self.goal_color]
        self.orientation = 0
        self.hero = gameOb(self.newPosition(0), 1, [0, 0, 1], None, 'hero')
        self.objects.append(self.hero)
        for i in range(self.sizeX - 1):
            bug = gameOb(self.newPosition(0), 1, self.goal_color, 1, 'goal')
            self.objects.append(bug)
        for i in range(self.sizeX - 1):
            hole = gameOb(self.newPosition(0), 1, self.other_color, 0, 'fire')
            self.objects.append(hole)
        state, s_big = self.renderEnv()
        self.state = state

        return state, s_big
    
    