import os
from baselines import logger
from baselines.common.vec_env import VecEnvWrapper
from gym.wrappers.monitoring import video_recorder


class VecVideoRecorder(VecEnvWrapper):
    """
    Wrap VecEnv to record rendered image as mp4 video.

    """

    def __init__(
            self,
            venv,
            directory,
            record_video_trigger,
            video_length=200):
        
        """
        Arguments
        -------------
        venv: 
            VecEnv to wrap.
        directory: 
            Where to save videos.
        record_video_trigger: 
            Function that defines when to start recording. The function takes 
                the current number of steps, and returns.
        video_length: 
            Length of recorded video. 

        """

        VecEnvWrapper.__init__(self, venv)
        self.record_video_trigger = record_video_trigger
        self.video_recorder = None

        self.directory = os.path.abspath(directory)
        if not os.path.exists(self.directory):
            os.mkdir(self.directory)
        
        self.file_prefix = "vecenv"
        self.file_infix = '{}'.format(os.getpid())
        self.step_id = 0
        self.video_length = video_length

        self.recording = False
        self.recorded_frames = 0
    
    