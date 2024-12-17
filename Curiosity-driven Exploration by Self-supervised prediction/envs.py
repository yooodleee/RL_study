from __future__ import print_function
from PIL import Image
from gym.spaces.box import Box
import gym.wrappers.monitoring
import numpy as np
import gym
from gym import spaces
import logging
import universe # docker-wsl 관련 설정(Python v3.5~3.6 권장)
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces import vnc_spaces
from universe.spaces.vnc_event import keycode
import env_wrapper
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
universe.configure_logging()


def create_env(env_id, client_id, remotes, **kwargs):
    if 'doom' in env_id.lower() or 'labyrinth' in env_id.lower():
        return create_doom(env_id, client_id, **kwargs)
    if 'mario' in env_id.lower():
        return create_mario(env_id, client_id, **kwargs)

    spec = gym.spec(env_id)
    if spec.tags.get('flashgames', False):
        return create_flash_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    else:
        # Assume atari.
        assert "." not in env_id    # universe environments have dots in names.
        return create_atari_env(env_id, **kwargs)

def create_doom(
    env_id, 
    client_id,
    envWrap=True,
    record=False,
    outdir=None,
    noLifeReward=False,
    acRepeat=0,
    **_,
):
    from ppaquette_gym_doom import wrappers
    if 'labyrinth' in env_id.lower():
        if 'single' in env_id.lower():
            env_id = 'ppaquette/LabyrinthSingle-v0'
        elif 'fix' in env_id.lower():
            env_id = 'ppaquette/LabyrinthManyFixed-v0'
        else:
            env_id = 'ppaquette/LabyrinthMany-v0'
    elif 'very' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayHomeFixed15-v0'
    elif 'sparse' in env_id.lower():
        env_id = 'ppaquette/DoomMyWayNomeFixed-v0'
    elif 'fix' in env_id.lower():
        if '1' in env_id or '2' in env_id:
            env_id = 'ppaquette/DoomMyWayHomeFixed' + str(env_id[-2:]) + '-v0'
        elif 'new' in env_id.lower():
            env_id = 'ppaquette/DoomMyWayHomeFixedNew-v0'
        else:
            env_id = 'ppaquette/DoomMyWayHomeFixed-v0'
    else:
        env_id = 'ppaquette/DoomMyWayHome-v0'
    
    # VizDoom workaround: Simultaneously launching multiple vizdoom processes
    # make program stuck, so use the global lock in multi-threading/processing
    client_id = int(client_id)
    time.sleep(client_id * 10)
    env = gym.make(env_id)
    modewrapper = wrappers.SetPlayingMode('algo')
    obwrapper = wrappers.SetResolution('160x120')
    acwrapper = wrappers.ToDiscrete('minimal')
    env = modewrapper(obwrapper(acwrapper(env)))
    # env = env_wrapper.MakeEnvDynamic(env) # to add stochasticity

    if record and outdir is not None:
        env = gym.wrappers.monitoring(env, outdir, force=True)
    
    if envWrap:
        fshape = (42, 42)
        frame_skip = acRepeat if acRepeat > 0 else 4
        env.seed(None)
        if noLifeReward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(env, skip=frame_skip, shape=fshape)
        env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif noLifeReward:
        env = env_wrapper.NoNegativeRewardEnv(env)
    
    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def create_mario(
    env_id,
    client_id,
    envWrap=True,
    record=False,
    outdir=None,
    noLifeReward=False,
    acRepeat=0,
    **_,
):
    import ppaquette_gym_super_mario 
    from ppaquette_gym_super_mario import wrappers
    if '-v' in env_id.lower():
        env_id = 'ppaquette/' + env_id
    else:
        env_id = 'ppaquette/SuperMarioBros-1-1v0'   # shape: (224, 256, 3)=(h,w,c)

    # Mario workaround: Simultaneously launching multiple vizdoom processes makes program stuck,
    # so use the global lock in multi-threading/multi-processing
    # see: https://github.com/ppaquette/gym-super-mario/tree/master/ppaquette_gym_super_mario
    client_id = int(client_id)
    time.sleep(client_id * 50)
    env = gym.make(env_id)
    modewrapper = wrappers.SetPlayingMode('algo')
    acwrapper = wrappers.ToDiscrete()
    env = modewrapper(acwrapper(env))
    env = env_wrapper.MarioEnv(env)

    if record and outdir is not None:
        env = gym.wrappers.monitoring(env, outdir, force=True)
    
    if envWrap:
        frame_skip = acRepeat if acRepeat > 0 else 6
        fshape = (42, 42)
        env.seed(True)
        if noLifeReward:
            env = env_wrapper.NoNegativeRewardEnv(env)
        env = env_wrapper.BufferedObsEnv(env, skip=frame_skip, shape=fshape, maxFrames=False)
        if frame_skip > 1:
            env = env_wrapper.SkipEnv(env, skip=frame_skip)
    elif noLifeReward:
        env = env_wrapper.NoNegativeRewardEnv(env)
    
    env = Vectorize(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    # env.close()   # TODO: think about where to put env.close !
    return env

def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registory
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    logger.info('create_flash_env(%s): keys=%s', env_id, keys)

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    env.configure(
        fps=5.0,
        remotes=remotes,
        start_timeout=15 * 60,
        client_id=client_id,
        vnc_spaces='go',
        vnc_kwargs={
            'encodinf': 'tight',
            'compress_level': 0,
            'fine_quality_level': 50,
            'subsample_level': 3
        }
    )
    
    return env

def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)

    logger.info("Connecting to remotes: %s", remotes)
    fps = env.metadata['video.frame_per_second']
    env.configures(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env

def create_atari_env(env_id, record=False, outidr=None, **_):
    env = gym.make(env_id)
    if record and outidr is not None:
        env = gym.wrappers.monitoring(env, outidr, force=True)
    env = Vectorize(env)
    env = AtariRescale42x42(env)
    env = DiagnosticsInfo(env)
    env = Unvectorize(env)
    return env

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)


class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1
    
    def _after_reset(self, observation):
        logger.info('Resetting environment logs')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation
    
    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()
        
        self._local_t += 1
        if info.get("Stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")
        
        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get("vectorized.episode_id", 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics.action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics.action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]
            
            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_stats.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]
    
        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('True Game terminating: env_episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []
        
        if 'distance' in info: to_log['distance'] = info['distance']    # mario
        if 'POSITION_X' in info:    # doom
            to_log['POSITION_X'] = info['POSITION_X']
            to_log['POSITION_Y'] = info['POSITION_Y']
        return observation, reward, done, to_log

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42
    # aren't close enough to the pixel boundary.
    frame = np.asarray(Image.fromarray(frame).resize((80, 80), resample=Image.BILINEAR).resize(
                        (42,42), resample=Image.BILINEAR))
    frame = frame.mean(2)   # take mean along channels
    frame = frame.astype(np.float32)
    frame += (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [42, 42, 1])
    
    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]


class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()
    
    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)
    
    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break        
        return action_n


class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-seperated list of keys

    For example,
        e=DiscreteTOFixedkeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]

    You can define a state with more than one key down by separating with spaces. For example,
        e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))
    
    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)
        
        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)
    
    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]


class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))
    
    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None for ob in observation_n]

def _process_frame_flash(frame):
    frame = np.array(Image.fromarray(frame).resize((200, 128), resample=Image.BILINEAR))
    frame = frame.mean(2).astype(np.float32)
    frame += (1.0 / 255.0)
    frame = np.reshape(frame, [128, 200, 1])
    return frame


class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [128, 200, 1])
    
    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]