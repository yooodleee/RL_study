import gym
from Brain import SACAgent
from Common import Play, Logger, get_params
import numpy as np
from tqdm import tqdm
import mujoco_py


def concat_state_latent(s, z_, n):
    z_one_hot = np.zeros(n)
    z_one_hot[z_] = 1
    return np.concatenate([s, z_one_hot])

