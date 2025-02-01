import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger


class Sampler(object):

    def start_worker(self):
        """
        Initialize the sampler, e.g., launching parallel workers if necessary.
        """
        raise NotImplementedError
    
    