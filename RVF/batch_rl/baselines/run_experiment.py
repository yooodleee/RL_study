
"""
Logged Runner.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dopamine.discrete_domains import run_experiment
import gin


@gin.configurable
class LoggedRunner(run_experiment.Runner):

    def run_experiment(self):
        super(LoggedRunner, self).run_experiment()
        # Log the replay buffer at the end
        self._agent.log_final_buffer()