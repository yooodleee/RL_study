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
    
    def reset(self):
        obs = self.venv.reset()

        self.start_video_recorder()

        return obs
    
    def start_video_recorder(self):
        self.close_video_recorder()

        base_path = os.path.join(
            self.directory,
            '{}.video.{}.video{:06}'.format(
                self.file_prefix,
                self.file_infix,
                self.step_id,
            )
        )
        self.video_recorder = video_recorder.VideoRecorder(
            env=self.venv,
            base_path=base_path,
            metadata={'step_id': self.step_id},
        )

        self.video_recorder.capture_frame()
        self.recorded_frames = 1
        self.recording = True
    
    