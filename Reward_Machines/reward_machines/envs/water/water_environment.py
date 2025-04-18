import gym
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.water.water_world import WaterWorld, WaterWorldParams, play



class WaterEnv(gym.Env):

    def __init__(self, state_file):
        params = WaterWorldParams(
            state_file, 
            b_radius=15,
            max_x=400,
            max_y=400,
            b_num_per_color=2,
            use_velocities=True,
            ball_disappear=False,
        )
        self.params = params
        
        self.action_space = spaces.Discrete(5)  # noop, up, right, down, left
        self.observation_space = spaces.Box(low=-2, high=2, shape=(52,), dtype=np.float)
        self.env = WaterWorld(params)
    

    def get_events(self):
        return self.env.get_true_propositions()
    

    def step(self, action):
        self.env.execute_action(action)
        obs = self.env.get_features()
        reward = 0  # all the reward comes from the RM
        done = False
        info = {}
        return obs, reward, done, info
    

    def reset(self):
        self.env.reset()
        return self.env.get_features()


###############################################################################
### MULTITASK
###############################################################################


class WaterRMEnv(RewardMachineEnv):

    def __init__(self, state_file):
        env = WaterEnv(state_file)
        rm_files = ["./envs/water/reward_machines/t%d.txt" % i for i in range(1, 11)]
        super().__init__(env, rm_files)
    

    def render(self, mode='human'):
        if mode == "human":
            play(self)
        else:
            raise NotImplementedError


class WaterRMEnvM1(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_1.pkl"
        super().__init__(state_file)


class WaterRMEnvM2(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_2.pkl"
        super().__init__(state_file)


class WaterRMEnvM3(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_3.pkl"
        super().__init__(state_file)


class WaterRMEnvM4(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_4.pkl"
        super().__init__(state_file)


class WaterRMEnvM5(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_5.pkl"
        super().__init__(state_file)


class WaterRMEnvM6(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_6.pkl"
        super().__init__(state_file)


class WaterRMEnvM7(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_7.pkl"
        super().__init__(state_file)


class WaterRMEnvM8(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_8.pkl"
        super().__init__(state_file)


class WaterRMEnvM9(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_9.pkl"
        super().__init__(state_file)


class WaterRMEnvM10(WaterRMEnv):

    def __init__(self):
        state_file = "./envs/water/maps/world_10.pkl"
        super().__init__(state_file)


###############################################################################
### SINGLE TASK
###############################################################################


class WaterRM10Env(RewardMachineEnv):

    def __init__(self, state_file):
        env = WaterEnv(state_file)
        rm_files = ["./envs/water/reward_machines/t10.txt"]
        super().__init__(env, rm_files)
    
    
    def render(self, mode="human"):
        if mode == "human":
            play(self)
        else:
            raise NotImplementedError


class WaterRM10EnvM0(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_0.pkl"
        super().__init__(state_file)


class WaterRM10EnvM1(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_1.pkl"
        super().__init__(state_file)


class WaterRM10EnvM2(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_2.pkl"
        super().__init__(state_file)


class WaterRM10EnvM3(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_3.pkl"
        super().__init__(state_file)


class WaterRM10EnvM4(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_4.pkl"
        super().__init__(state_file)


class WaterRM10EnvM5(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_5.pkl"
        super().__init__(state_file)


class WaterRM10EnvM6(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_6.pkl"
        super().__init__(state_file)


class WaterRM10EnvM7(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_7.pkl"
        super().__init__(state_file)


class WaterRM10EnvM8(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_8.pkl"
        super().__init__(state_file)


class WaterRM10EnvM9(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_9.pkl"
        super().__init__(state_file)


class WaterRM10EnvM10(WaterRM10Env):

    def __init__(self):
        state_file = "./envs/water/maps/world_10.pkl"
        super().__init__(state_file)