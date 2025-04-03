import os
import tempfile


import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np


import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds


from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput


from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func



class ActWrapper(object):

    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None
    

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_param)
        sess = tf.compat.v1.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)
            
            zipfile.ZipFile(arc_path, "r", zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))
        
        return ActWrapper(act, act_params)
    

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)
    

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None
    

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")
        
        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, "w") as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)
    

    def save(self, path):
        save_variables(path)


