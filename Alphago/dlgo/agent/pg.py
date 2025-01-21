import numpy as np
from keras import optimizers    # import SGD

from dlgo.agent.base import Agent
from dlgo.agent.helpers import is_point_an_eye
from dlgo import encoders
from dlgo import goboard
from dlgo import kerasutil


def prepare_experience_data(
        experience,
        board_width,
        board_height):
    
    experience_size = experience.actions.shape[0]
    target_vectors = np.zeros(
        (
            experience_size,
            board_width * board_height,
        )
    )
    for i in range(experience_size):
        action = experience.actions[i]
        reward = experience.rewards[i]
        target_vectors[i][action] = reward
    
    return target_vectors


