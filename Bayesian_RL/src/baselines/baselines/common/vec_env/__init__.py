from abc import ABC, abstractmethod
from baselines.common.title_images import title_images


class AlreadySteppingError(Exception):
    """
    Raised when an asynchronous step is running while
        step_async() is called again.

    """

    def __init__(self):
        msg = 'already running an async step'
        Exception.__init__(self, msg)
    

class NotSteppingError(Exception):
    """
    Raised when an asynchronous step is not running but
        step_wait() is called.

    """

    def __init__(self):
        msg = 'not running an async step'
        Exception.__init__(self, msg)


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
        each observation becomes an batch of observations, and expected
        action is a batch of actions to be applied per-environment.

    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(
            self,
            num_envs,
            observation_space,
            action_space):
        
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
    
    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
            observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
            be cancelled and step_wait() should not be called
            untill step_async() is invoked again.

        """
        pass

    