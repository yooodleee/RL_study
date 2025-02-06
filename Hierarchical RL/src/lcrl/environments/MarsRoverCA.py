
import os
import random
import numpy as np


from matplotlib.image import imread



class MarsRover:
    """
    An MDP whose labels depend on a baackground image.


    Attributes:
    ----------------
        image: (string)
            path to a layout file.


    Methods:
    ------------------
        reset():
            resets the MDP state, including ghosts and agent.
        step(action):
            changes the state of the MDP upon executing an action,
            where the action set is { right, up, left, down, stay }.
        state_label(state):
            outputs the label of input state.
    """

    def __init__(
            self,
            image='marsrover_1.png',
            initial_state=None):
        
        self.background = imread(
            os.path.join(os.path.abspath(__file__)),
            'layouts',
            image
        )
        
        self.width = self.background.shape[0]
        self.height = self.background.shape[1]
        self.labels = self.background.__array__()
        self.initial_state = initial_state

        if self.initial_state is None:
            self.initial_state = np.array(
                [60, 100], dtype=np.float32
            )
        
        self.current_state = self.initial_state
        
        # range for the sine of action angle direction.
        self.action_space = [1, -1]
    

    def reset(self):
        self.current_state = self.initial_state.copy()
    

    def step(self, action):
        """
        agent movement dynamics:
            stochasticity
        """

        traversed_distance = 4 + random.random()
        noise = np.array(
            [
                random.uniform(-0.1, 0.5),
                random.uniform(-0.1, 0.5)
            ]
        )
        next_state = self.current_state \
                    + noise \
                    + np.append(
                        traversed_distance * np.sin(action[0] * np.pi),
                        traversed_distance * np.cos(action[0] * np.pi)
                    )
        
        # check for boundary violations
        if next_state[0] > self.width - 1:
            next_state[0] = self.width - 1
        
        if next_state[1] > self.height - 1:
            next_state[1] = self.height - 1
        
        if next_state[0] < 0:
            next_state[0] = 0
        
        if next_state[1] < 0:
            next_state[1] = 0
        

        # update current state
        self.current_state = next_state
        return next_state
    

    