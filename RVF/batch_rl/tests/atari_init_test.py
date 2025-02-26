
"""A simple test for validating that the Atari env initializers."""

import datetime
import os
import shutil


from absl import flags
from batch_rl.baselines import train
import tensorflow as tf # compat to version 1.x


FLAGS = flags.FLAGS



class AtariInitTest(tf.test.TestCase):

    def setUp(self):
        super(AtariInitTest, self).setUp()
        FLAGS.base_dir = os.path.join(
            '/tmp/batch_rl_tests',
            datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S')
        )
        FLAGS.gin_files = ['batch_rl/baselines/configs/dqn.gin']

        # `num_iterations` set to zero to prevent runner execution.
        FLAGS.gin_bindings = [
            'Runner.num_iterations=0',
            'WrappedReplayBuffer.replay_capacity = 100' # To prevent OOM.
        ]
        FLAGS.alsologotostderr = True
    

    def test_atari_init(self):
        """Test that a DQN agent is initialized."""
        train.main([])
        shutil.rmtree(FLAGS.base_dir)



if __name__ == '__main__':
    tf.test.main()