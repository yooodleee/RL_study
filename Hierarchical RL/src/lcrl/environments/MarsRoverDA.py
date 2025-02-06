
import os
import random
import numpy as np

from matplotlib.image import imread



class MarsRover:
    """
    An MDP whose labels depend on a background image.


    Attributes:
    ------------------
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
            os.path.join(
                os.path.dirname(
                    os.path.abspath(__file__)
                ),
                'layouts',
                image
            )
        )

        self.width = self.background.shape[0]
        self.height = self.background.shape[1]
        self.labels = self.background.__array__()
        self.initial_state = initial_state

        if self.initial_state is None:
            self.initial_state = np.array(
                [35, 25], dtype=np.float32
            )
        
        self.current_state = self.initial_state

        # directional acts
        self.action_space = [
            "right", "up", "left", "down", "stay"
        ]


    