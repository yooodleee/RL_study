from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from algos.batch_polopt import BatchPolopt
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


class PPO(BatchPolopt):
    """
    Proximal Policy Optimization.
    """

    def __init__(
            self,
            clip_lr=0.3,
            increase_penalty_factor=2,
            decrease_penalty_factor=0.5,
            min_penalty=1e-3,
            max_penalty=1e-6,
            entropy_bonus_coeff=0.,
            gradient_clipping=40.,
            log_loss_kl_before=True,
            log_loss_kl_after=True,
            use_kl_penalty=False,
            initial_kl_penalty=1.,
            use_line_search=True,
            max_backtracks=10,
            backtrack_ratio=0.5,
            optimizeer=None,
            step_size=0.01,
            min_n_epochs=2,
            adaptive_learning_rate=False,
            max_learning_rate=1e-3,
            min_learning_rate=1e-5,
            **kwargs):
        
        self.clip_lr = clip_lr
        self.increase_penalty_factor = increase_penalty_factor
        self.decrease_penalty_factor = decrease_penalty_factor
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.entropy_bonus_coeff = entropy_bonus_coeff
        self.gradient_clipping = gradient_clipping
        self.log_loss_kl_before = log_loss_kl_before
        self.log_loss_kl_after = log_loss_kl_after
        self.use_kl_penalty = use_kl_penalty
        self.initial_kl_penalty = initial_kl_penalty
        self.use_line_search = use_line_search
        self.max_backtracks = max_backtracks
        self.backtrack_ratio = backtrack_ratio
        self.step_size = step_size
        self.min_n_epochs = min_n_epochs
        self.adaptive_learning_rate = adaptive_learning_rate
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        if optimizeer is NOne:
            optimizeer = AdamOptimizer()
        self.optimizer = optimizeer
        super().__init__(**kwargs)

    