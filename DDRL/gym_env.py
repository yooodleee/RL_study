"""
gym environment processing components.
"""

import os
import datetime
import gym.spaces
import gym.wrappers
import gym.wrappers
import gym.wrappers
import gym.wrappers
import gym.wrappers.clip_action
import gym.wrappers.time_limit
import numpy as np
import cv2
import logging
import gym
from gym.spaces import Box
from collections import deque
from pathlib import Path

# pylint: disable=import-error
from . import types as types_lib


# A simple list of classic env names.
CLASSIC_ENV_NAMES = [
    'CartPole-v1',
    'LunarLander-v2',
    'MontainCar-v0',
    'Acrobot-v1',
]


def unwrap(env):
    if hasattr(env, 'unwrapped'):
        return env.unwrapped
    elif hasattr(env, 'env'):
        return unwrap(env.env)
    elif hasattr(env, 'leg_env'):
        return unwrap(env.leg_env)
    else:
        return env


class NoopReset(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(
        self, env, noop_max=30
    ):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
    
    def reset(self, **kwargs):
        """
        Do no-op action for a number of steps in [1, noop_max].
        """
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(
                1, self.noop_max + 1
            )   # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        return self.env.step(action)


class FireOnReset(gym.Wrapper):
    """
    Take fire action on reset for environments like Breakout.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs):
        self.env.reset(**kwargs)

        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.render(**kwargs)

        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        return self.env.step(action)


class StickyAction(gym.Wrapper):
    """
    Repeats the last action with epsilon (default 0.25) probability.
    """

    def __init__(self, env, eps=0.25):
        gym.Wrapper.__init__(self, env)
        self.eps = eps
        self.last_action = 0
    
    def step(self, action):
        if np.random.uniform() < self.eps:
            action = self.last_action
        
        self.last_action = action
        return self.env.step(action)
    
    def reset(self, **kwargs):
        self.last_action = 0
        return self.env.reset(**kwargs)


class LifeLoss(gym.Wrapper):
    """
    Adds boolean key 'loss_life' into the info dict, but only reset on true 
        game over.
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_terminated = True
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_terminated = done

        # Check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()

        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            info['loss_life'] = True
        else:
            info['loss_life'] = False
        
        self.lives = lives
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
            and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_terminated:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.lives()
        return obs
    

class MaxAndSkip(gym.Wrapper):
    """
    Return only every `skip`-th frame
    """

    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, dtype=np.uint8
        )
        self._skip = skip
    
    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.
        """
        total_reward = 0.0
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grayscale image as done in the Nature paper.
    """

    def __init__(
        self,
        env, 
        width=84,
        height=84,
        grayscale=True,
    ):
        super().__init__(env)

        assert self.observation_space.dtype \
                == np.unit8 and len(self.observation_space.shape) \
                == 3
        
        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.unit8,
        )
    
    def observation(self, obs):
        # pylint: disable=no-member

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs,
            (self.frame_width, self.frame_height),
            interpolation=cv2.INTER_AREA, 
        )
        # pylint: disable=no-member

        if self.grayscale:
            obs = np.expand_dims(obs, -1)
        
        return obs


class FrameStack(gym.Wrapper):
    """
    Stack k last frames.
    
    Returns lazy array, which is much more memory efficient.
    See also
    --------
    baselines.common.atari_wrappers.LazyFrames
    """

    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(shape[:-1] + (shape[-1] * k)),
            dtype=env.observation_space.dtype,
        )
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    """
    This object ensures that common frames between the observations are
        only stored once. It exists purely to optimize memory usage which 
        can be huge for DQN's 1M frames replay buffers.
    This object should only be converted to numpy array before being passed
        to the model. You'd not believe how complex the previous solution was.
    """

    def __init__(self, frames):
        self.dtype = frames[0].dtype
        self.shape = (
            frames[0].shape[0],
            frames[0].shape[1],
            len(frames)
        )
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out
    
    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out
    
    def __len__(self):
        return len(self._force())
    
    def __getitem__(self, i):
        return self._force()[i]
    
    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]
    
    def frame(self, i):
        return self._force()[..., i]


class ScaleFrame(gym.ObservationWrapper):
    """
    Scale frame by divide 255.
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
    
    def observation(self, obs):
        # Carefull! This undoes the memory optimization, use
        # with smaller replay buffer only.
        return np.array(obs).astype(np.float32) / 255.0


class VisitedRoomInfo(gym.Wrapper):
    """
    Add number of unique visited rooms to the info dictionary.
    For Atari games like MontezumaRevenge and Pitfall.
    """

    def __init__(self, env, room_address):
        gym.Wrapper.__init__(self, env)
        self.room_address = room_address
        self.visited_rooms = set()
    
    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])
    
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            info['episode_visited_rooms'] = len(self.visited_rooms)
            self.visited_rooms.clear()
        return obs, rew, done, info


class ObscureObservation(gym.ObservationWrapper):
    """
    Make the environment POMDP by obscure the state with probability epsilon.
    this should be used before frame stack.
    """

    def __init__(self, env, epsilon: float = 0.0):
        super().__init__(env)

        if not 0.0 <= epsilon < 1.0:
            raise ValueError(
                f'Expect obscure epsilon should be between [0.0, 1), '
                f'got {epsilon}'
            )
        self._eps = epsilon
    
    def observation(self, obs):
        if self.env.unwrapped.np_random.random() <= self._eps:
            obs = np.zeros_like(
                obs, dtype=self.observation_space.dtype
            )
        return obs


class ClipRewardWithBound(gym.RewardWrapper):
    """
    Clip reward to in the range [-bound, bound]
    """

    def __init__(self, env, bound):
        super().__init__(env)
        self.bound = bound
    
    def reward(self, reward):
        return None if reward is None \
                else max(min(reward, self.bound), -self.bound)


class ObservationChannelFirst(gym.ObservationWrapper):
    """
    Make observation image channel first, this is for PyTorch only.
    """

    def __init__(self, env, scale_obs):
        super().__init__(env)
        old_shape = env.observation_space.shape
        new_shape = (
            old_shape[-1], old_shape[0], old_shape[1]
        )
        _low, _high = (0.0, 255) if not scale_obs else (0.0, 1.0)
        new_dtype = env.observation_space.dtype \
                    if not scale_obs else np.float32
        self.observation_space = Box(
            low=_low,
            high=_high,
            shape=new_shape,
            dtype=new_dtype,
        )
    
    def observation(self, obs):
        # Permute [H, W, C] array to in the range [C, H, W]
        # return np.transpose(observation, axes=(2, 0, 1)).astype(self.observation_space.dtype)
        obs = np.asarray(
            obs, dtype=self.observation_space.dtype
        ).transpose(2, 0, 1)
        # Make sure it's C-contiguous for compress state
        return np.ascontiguousarray(
            obs, dtype=self.observation_space.dtype
        )


class ObservationToNumpy(gym.ObservationWrapper):
    """
    Make the observation into numpy ndarrays.
    """

    def observation(self, obs):
        return np.asarray(
            obs, dtype=self.observation_space.dtype
        )


class ClipObservationWithBound(gym.ObservationWrapper):
    """
    Make the observation into [-max_abs_value, max_abs_value].
    """

    def __init__(
        self, env, max_abs_value
    ):
        super().__init__(env)
        self._max_abs_value = max_abs_value
    
    def observation(self, obs):
        return np.clip(
            obs, -self._max_abs_value, self._max_abs_value
        )


class RecordRawReward(gym.Wrapper):
    """
    This wrapper will add non-clipped/unscaled raw reward to the info dict.
    """

    def step(self, action):
        """
        Take action and add non-clipped/unscaled raw reward to the info dict.
        """

        obs, reward, done, info = self.env.step(action)
        info['raw_reward'] = reward

        return obs, reward, done, info


def create_atari_environment(
    env_name: str,
    seed: int = 1,
    frame_skip: int = 4,
    frame_stack: int = 4,
    frame_height: int = 84,
    frame_width: int = 84,
    noop_max: int = 30,
    max_episode_steps: int = 108000,
    obscure_epsilon: float = 0.0,
    terminal_on_life_loss: bool = False,
    clip_reward: bool = True,
    sticky_action: bool = True,
    scale_obs: bool = False,
    channel_first: bool = True,
)-> gym.Env:
    """
    Process gym env for Atari games according to the Nature DQN paper.

    Args:
        env_name: the environment name without 'NoFrameskip' and version.
        seed: seed the runtime.
        frame_skip: the frequency at which the agent experiments the game,
            the environment will also repeat action.
        frame_stack: stack n last frames.
        frame_height: height of the resized frame.
        frame_width: width of the resized frame.
        noop_max: maximum number of no-ops to apply at the beginning
            of each episode to reduce determinism. These no-ops are appled at a
            low-level, before frame skipping.
        max_episode_steps: maximum steps for an episode.
        obscure_epsilon: with epsilon probability [0.0, 1.0), obscure the state to make it POMDP.
        terminal_on_life_loss: if True, mark end of game when loss a life, default off.
        clip_reward: clip reward in the range of [-1, 1], default on.
        sticky_action: if True, randomly re-use last action with 0.25 probability, default on.
        scale_on: scale the frame by divide 255, turn this on may require 4-5x more RAM when using experience replay, default off.
       channel_first: if True, change observation image from shape [H, W, C] to in the range [C, H, W], this is for PyTorch only, default on. 
    
    Returns:
        preprocessed gym.Env for Atari games.
    """

    if 'NoFrameskip' in env_name:
        raise ValueError(
            f'Environment name should not include NoFrameskip, got {env_name}'
        )
    
    env = gym.make(f'{env_name}NoFrameskip-v4')
    env.seed(seed)

    # Change TimeLimit wrapper to 108,000 steps (30 min) as default in the
    # literature instead of OpenAI Gym's default of 100,000 steps.
    env = gym.wrappers.TimeLimit(
        env.env, max_episode_steps=None if max_episode_steps <= 0 else max_episode_steps
    )

    if noop_max > 0:
        env = NoopReset(env, noop_max=noop_max)
    if sticky_action:
        env = StickyAction(env)
    if frame_skip > 0:
        env = MaxAndSkip(env, skip=frame_skip)
    
    # Obscure observation with obscure_epsilon probability
    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)
    if terminal_on_life_loss:
        env = LifeLoss(env)
    
    env = ResizeAndGrayscaleFrame(
        env, width=frame_width, height=frame_height
    )

    if scale_obs:
        env = ScaleFrame(env)

    if clip_reward:
        env = RecordRawReward(env)
        env = ClipRewardWithBound(env)
    
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if channel_first:
        env = ObservationChannelFirst(env, scale_obs)
    else:
        # This is required as LazeFrame object is not numpy.array.
        env = ObservationToNumpy(env)
    
    if 'Montezuma' in env_name or 'pitfall' in env_name:
        env = VisitedRoomInfo(
            env, room_address=3 if 'Montezuma' in env_name else 1
        )

    return env


def create_classic_environment(
    env_name: str,
    seed: int = 1,
    max_abs_reward: int = None,
    obscure_epsilon: float = 0.0,
)-> gym.Env:
    """
    Process gym env for classic control tasks like CartPole, Lunarlander, MountainCar

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.
        max_abs_reward: clip reward in the range of [-max_abs_reward, max_abs_reward], default off.
        obscure_epsilon: with epsilon probability [0.0, 1.0) obscure the state to make it POMDP.

    Returns:
        gym.Env for classic control tasks.
    """

    env = gym.make(env_name)
    
    # Clip reward to max absolute reward bound
    if max_abs_reward is not None:
        env = RecordRawReward(env)
        env = ClipRewardWithBound(env, abs(max_abs_reward))
    
    # Obscure observation with obscure_epsilon_probability
    if obscure_epsilon > 0.0:
        env = ObscureObservation(env, obscure_epsilon)
    
    return env


def create_continuous_environment(
    env_name: str,
    seed: int = 1,
    max_abs_obs: int = 10,
    max_abs_reward: int = 10,
)-> gym.Env:
    """
    Process gym env for classic robotic control tasks like Humanoid, Ant.

    Args:
        env_name: the environment name with version attached.
        seed: seed the runtime.
        max_abs_obs: clip observation in the range of [-max_abs_obs, max_abs_obs],
            defualt 10.
        max_abs_reward: clip reward in the range of [-max_abs_reward, max_abs_reward],
            default 10.

    Returns:
        gym.Env for classic roboric control tasks.
    """

    env = gym.make(env_name)

    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)

    # Optionally clipping the observation and rewards.
    # Notice using lambda function does not work with python multiprocessing
    # env = gym.wrappers.TransformObservation(env, lambda reward: np.clip(obs, -max_abs_obs, max_abs_obs))
    # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -max_abs_reward, max_abs_reward))
    env = ClipObservationWithBound(env, max_abs_obs)
    env = ClipRewardWithBound(env, max_abs_reward)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    
    return env


