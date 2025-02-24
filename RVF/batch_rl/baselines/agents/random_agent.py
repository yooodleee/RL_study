
"""
Random agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.agents.dqn import dqn_agent
import numpy as np

import gin


@gin.configurable
class RandomAgent(dqn_agent.DQNAgent):
    """
    Random Agent.
    """

    def __init__(
            self,
            sess,
            num_actions,
            replay_log_dir,
            **kwargs):
        
        """
        This maintains all the DQN default argument values.
        """
        self._replay_log_dir = replay_log_dir
        super(RandomAgent, self).__init__(sess, num_actions, **kwargs)
    
    def step(self, reward, observation):
        """
        Returns a random action.
        """
        return np.random.randint(self.num_actions)
    
    def log_final_buffer(self):
        pass