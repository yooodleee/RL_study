from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from sandbox.rocky.tf.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class NPO(BatchPolopt):
    """
    Neural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            **kwargs):
        
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optmizer = optimizer
        self.step_size = step_size
        super(NPO, self).__init__(**kwargs)
    
    