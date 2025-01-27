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
    
    def moveChar(self, action):
        # 0 - up, 1 - down, 2 - left, 3 - right, 4 - 90 counter-clockwise, 5 - 90 clockwise
        hero = self.objects[0]
        blockPositions = [[-1, -1]]
        for ob in self.objects:
            if ob.name == 'block': blockPositions.append([ob.x, ob.y])
        blockPositions = np.array(blockPositions)
        heroX = hero.x
        heroY = hero.y
        penalize = 0
        if action < 4:
            if self.orientation == 0:
                direction = action
            if self.orientation == 1:
                if action == 0: direction = 1
                elif action == 1: direction = 0
                elif action == 2: direction = 3
                elif action == 3: direction = 2
            if self.orientation == 2:
                if action == 0: direction = 3
                elif action == 1: direction = 2
                elif action == 2: direction = 0
                elif action == 3: direction = 1
            if self.orientation == 3:
                if action == 0: direction = 2
                elif action == 1: direction = 3
                elif action == 2: direction = 1
                elif action == 3: direction = 0
            
            if direction == 0 and hero.y >= 1 and [hero.x, hero.y - 1] not in blockPositions.tolist():
                hero.y -= 1
            if direction == 1 and hero.y <= self.sizeY-2 and [hero.x, hero.y + 1] not in blockPositions.tolist():
                hero.y += 1
            if direction == 2 and hero.x >= 1 and [hero.x - 1, hero.y] not in blockPositions.tolist():
                hero.x -= 1
            if direction == 3 and hero.x <= self.sizeX-2 and [hero.x + 1, hero.y] not in blockPositions.tolist():
                hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero

        return penalize
    
    