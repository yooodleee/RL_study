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
    

