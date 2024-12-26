import numpy as np

from multiprocessing import Pipe, connection
from threading import Thread
from time import time
from logging import getLogger

from alpha_zero.agent.model import AlphaModel
from alpha_zero.config import Config

from alpha_zero.lib.model_helper import (
    reload_newest_next_generation_model_if_changed,
    load_best_model_weight,
    save_as_best_model,
    reload_best_model_weight_if_changed
)
import tensorflow as tf


logger = getLogger(__name__)



class AlphaModelAPI:
    def __init__(self, config: Config, agent_model):
        """
        params:
            config
            alpha_zero.agent.model.alphaModel agent_model
        """
        self.config = config
        self.agent_model = agent_model
    
    def predict(self, x):
        assert x.ndim in (3, 4)
        assert x.shape == (2, 8, 8) or x.shape[1:] == (2, 8, 8)
        orig_x = x
        if x.ndim == 3:
            x = x.reshape(1, 2, 8, 8)
        
        policy, value = self._do_predict(x)

        if orig_x.ndim == 3:
            return policy[0], value[0]
        else:
            return policy, value
    
    def _do_predict(self, x):
        return self.agent_model.model.predict_on_batch(x)



class MultiProcessAlphaModelAPIServer:
    # https://github.com/Akababa/Chess-Zero/blob/nohistory/src/chess_zero/agent/api_chess.py

    def __init__(self, config: Config):
        self.config = config
        self.model = None   # type: AlphaModel