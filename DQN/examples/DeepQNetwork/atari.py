import atari_py.import_roms
import numpy as np
import os
import threading
import cv2
import gym
import six

import atari_py

# ROM 경로 설정
# ROM_URL="https://github.com/openai/atari-py/tree/gdb/atari_py/atari_roms"
ROM_URL = "C:/Users/dhals_zn0ga5j/ROMS"

# 경로 유효성 확인
if os.path.exists(ROM_URL):
    print("경로가 유효합니다!")
else:
    print("경로를 확인하세요!")

# ROM 불러오기
try:
    atari_py.import_roms(ROM_URL)
    print("ROM이 성공적으로 불러와졌습니다!")
except Exception as e:
    print(f"오류 발생: {e}")

from ale_py import ALEInterface, Action, LoggerMode
from gym import spaces

from tensorpack.utils import logger, execute_only_once, get_rng
from tensorpack.utils.fs import get_dataset_path

__all__=['AtariPlayer']

_ALE_LOCK=threading.Lock()


class AtariPlayer(gym.Env):
    """
    A wrapper for ALE emulator, with configurations to mimic DeepMind DQN settings.

    Info:
        score: the accumulated reward in the current game
        gameOver: True when the current game is Over
    """

    def __init__(self, rom_file, viz=0,
                 frame_skip=4, nullop_start=30,
                 live_lost_as_eoe=True, max_num_frames=0,
                 grayscale=True):
        """
        Args:
            rom_file; path to the rom
            frame_skip: skip every k frames and repeat the action
            viz: Visualization to be done.
                Set to 0 to disable.
                Set to a positive number to the delay between frames to show.
                Set to a string to be a directory to store frames.
            nullop_start: start with random number of null ops.
            live_losts_as_eoe: consider lost of lives as end of episode. Useful for training.
            max_num_frames: maximum number of frames per episode.
            grayscale (bool): if True, return 2D image. Otherwise return HWC image.
        """
        super(AtariPlayer, self).__init__()
        rom_file="breakout.bin"
        if not os.path.isfile(rom_file) and '/' not in rom_file:
            rom_file=get_dataset_path('ROMS', rom_file)    # get_dataset_path 수정 확인 ('atari_rom' -> 'ROMS')
        assert os.path.isfile(rom_file), \
            "ROM {} not found. Please download at {}".format(rom_file, ROM_URL)
        
        try:
            ALEInterface.setLoggerMode(LoggerMode.Error)
        except AttributeError:
            if execute_only_once():
                logger.warn("You're not using latest ALE")
        
        with _ALE_LOCK:
            self.ale=ALEInterface()
            self.rng=get_rng(self)
            self.ale.setInt(b"random_seed", self.rng.randint(0, 300000))
            self.ale.setInt(b"max_num_frames_per_episode", max_num_frames)

            self.ale.setInt(b"frame_skip", 1)
            self.ale.setBool(b'color_averaging', False)
            # manual.pdf wuggests otherwise
            self.ale.setFloat(b'repeat_action_probabiliaty', 0.0)

            # viz setup
            if isinstance(viz, six.string_types):
                assert os.path.isdir(viz), viz
                self.ale.setString(b'record_screen_dir', viz)
                viz=0
            if isinstance(viz, int):
                viz=float(viz)
            self.viz=viz
            if self.viz and isinstance(self.viz, float):
                self.windowname=os.path.basename(rom_file)
                cv2.namedWindow(self.windowname)
            
            self.ale.loadROM(rom_file.encode('utf-8'))
        self.width, self.height=self.ale.getScreenDims()
        self.actions=self.ale.getMinimalActionSet()

        self.live_lost_as_eoe=live_lost_as_eoe
        self.frame_skip=frame_skip
        self.nullop_start=nullop_start

        self.action_space=spaces.Discrete(len(self.actions))
        self.grayscale=grayscale
        shape=(self.height, self.width) if grayscale else (self.height, self.width, 3)
        self.observation_space=spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8
        )
        self._restart_episode()
    
    def get_action_meanings(self):
        keys=Action.__members__.values()
        values=Action.__members__.keys()
        mapping=dict(zip(keys, values))
        return [mapping[action] for action in self.actions]
    
    def _grab_raw_image(self):
        """
        :returns: the current 3-channel image
        """
        m=self.ale.getScreenRGB()
        return m.reshape((self.height, self.width, 3))
    
    def _current_state(self):
        """
        :returns: a gray-scale (h, w) unit8 image
        """
        ret=self._grab_raw_image()
        # max-pooled over the last screen
        ret=np.maximum(ret, self.last_raw_screen)
        if self.viz:
            if isinstance(self.viz, float):
                cv2.imshow(self.windowname, ret)
                cv2.waitKey(int(self.viz * 1000))
        if self.grayscale:
            # 0.2999, 0.587.0.114. same as rgb2y in torch/image
            ret=cv2.cvtColor(ret, cv2.COLOR_RGB2GRAY)
        return ret.astype('unit-8') # to save some memory
    
    def _restart_episode(self):
        with _ALE_LOCK:
            self.ale.reset_game()
        
        # random null-ops start
        n=self.rng.randint(self.nullop_start)
        self.last_raw_screen=self._grab_raw_image()
        for k in range(n):
            if k == n - 1:
                self.last_raw_screen=self._grab_raw_image()
            self.ale.act(0)
    
    def reset(self):
        if self.ale.game_over():
            self._reset_episode()
        return self._current_state()
    
    def render(self, *args, **kwargs):
        pass # visualization for this env is through the viz=argument when creating the player

    def step(self, act):
        oldlives=self.ale.lives()
        r=0
        for k in range(self.frame_skip):
            if k == self.frame_skip - 1:
                self.last_raw_screen=self._grab_raw_image()
            r += self.ale.act(self.actions[act])
            newlives=self.ale.lives()
            if self.ale.game_over() or \
                    (self.live_lost_as_eoe and newlives < oldlives):
                break
        
        isOver=self.ale.game_over()
        if self.live_lost_as_eoe:
            isOver=isOver or newlives < oldlives
        
        info={'ale.lives': newlives}
        return self._current_state(), r, isOver, info


if __name__ == "__main":
    import sys

    a=AtariPlayer(sys.argv[1], viz=0.03)
    num=a.action_space.n
    rng=get_rng(num)
    while True:
        act=rng.choice(range(num))
        state, reward, isOver, info=a.step(act)
        if isOver:
            print(info)
            a.reset()
        print("Reward:", reward)