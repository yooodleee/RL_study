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
    
    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        : param itr: Iteration number.
        : return: A list of paths.
        """
        raise NotImplementedError
    
    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated
            tensors) based on the collected paths.
        
        params
        ---------
        itr:
            Iteration number.
        paths:
            A list of collected paths.

        return
        ----------
        Processed sample data.
        """
        raise NotImplementedError
    
    