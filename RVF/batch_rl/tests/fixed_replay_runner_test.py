
"""End to end tests FixedReplayRunner."""

import datetime
import os
import shutil


from absl import flags
from batch_rl.fixed_replay import train
import tensorflow as tf # compat to version 1.x


FLAGS = flags.FLAGS



class FixedReplayRunnerIntegrationTest(tf.test.TestCase):
    """Test for Atari env with various agents."""

    def setUp(self):
        super(FixedReplayRunnerIntegrationTest, self).setUp()
        
        FLAGS.base_dir = os.path.join(
            '/tmp/batch_rl_tests',
            datetime.datetime.utcnow().strftime('run_%Y_%m_%d_%H_%M_%S')
        )
        self._checkpoint_dir = os.path.join(FLAGS.base_dir, 'checkpoints')
        self._logging_dir = os.path.join(FLAGS.base_dir, 'logs')
    

    def quickFixedReplayREMFlags(self):
        """Assign flags for a quick run of FixedReplay agent."""

        FLAGS.gin_bindings = [
            "create_runner.schedule='continuous_train_and_eval'",
            'FixedReplayRunner.training_steps=100',
            'FixedReplayRunner.evaluation_step=10',
            'FixedReplayRunner.num_iterations=1',
            'FixedReplayRunner.max_steps_per_episode=100',
        ]
        FLAGS.alsologtostderr = True
        FLAGS.gin_files = ['batch_rl/fixed_replay/configs/rem.gin']
        FLAGS.agent_name = 'multi_head_dqn'
    

    def verifyFilesCreated(self, base_dir):
        """Verify that files have been created."""

        # Check checkpoint files.
        self.assertTrue(os.path.exists(os.path.join(self._checkpoint_dir, 'ckpt.0')))
        self.assertTrue(os.path.exists(os.path.join(self._checkpoint_dir, 'checkpoint')))
        self.assertTrue(os.path.exists(os.path.join(self._checkpoint_dir, 'sentinel_checkpoint_complete.0')))

        # Check log files.
        self.assertTrue(os.path.exists(os.path.join(self._logging_dir, 'log_0')))
    

    def testIntegrationFixedReplayREM(self):
        """Test the FixedReplayMultiHeadDQN agent."""

        assert FLAGS.replay_dir is not None, 'Please provide a replay directory'

        tf.compat.v1.logging.info('###### Training the REM agent #####')
        tf.compat.v1.logging.info('###### REM base_dir: {}'.format(FLAGS.base_dir))
        tf.compat.v1.logging.info('###### replay_dir: {}'.format(FLAGS.replay_dir))
        
        self.quickFixedReplayREMFlags()
        train.main([])
        self.verifyFilesCreated(FLAGS.base_dir)
        shutil.rmtree(FLAGS.base_dir)



if __name__ == '__main__':
    tf.test.main()