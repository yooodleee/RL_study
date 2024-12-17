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

    keys = []