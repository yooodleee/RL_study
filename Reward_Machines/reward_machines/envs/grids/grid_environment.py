import gym
import random
from gym import spaces
import numpy as np
from reward_machines.rm_environment import RewardMachineEnv
from envs.grids.craft_world import CraftWorld
from envs.grids.office_world import OfficeWorld
from envs.grids.value_iteration import value_iteration


class GridEnv(gym.Env):

    def __init__(self, env):
        self.env = env
        N, M = self.env.map_heigth, self.env.map_width
        self.action_space = spaces.Discrete(4)  # up, right, down, left
        self.observation_space = spaces.Box(
            low=0, high=max([N, M]), shape=(2,), dtype=np.uint8
        )
    
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
    
    def show(self):
        self.env.show()
    
    def get_model(self):
        return self.env.get_model()


class GridRMEnv(RewardMachineEnv):

    def __init__(self, env, rm_files):
        super().__init__(env, rm_files)
    
    def render(self, mode='human'):
        if mode == 'human':
            # commands
            str_to_action = {"w": 0, "d": 1, "s": 2, "a": 3}

            # play the game
            done = True
            while True:
                if done:
                    print(
                        "New episode -----------------------------"
                    )
                    obs = self.reset()
                    print(
                        "Current task: ", self.rm_files[self.current_rm_id] 
                    )
                    self.env.show()
                    print("Features: ", obs)
                    print("RM state: ", self.current_u_id)
                    print("Events: ", self.env.get_events())
                
                print("\nAction? (WASD keys or q to quite) ", end="")
                a = input()
                print()
                if a == 'q':
                    break
                # Executing action
                if a in str_to_action:
                    obs, rew, done, _ = self.step(str_to_action[a])
                    self.env.show()
                    print("Features: ", obs)
                    print("Reward: ", rew)
                    print("RM state: ", self.current_u_id)
                    print("Events: ", self.env.get_events())
                else:
                    print("Forbidden action")
        else:
            raise NotImplementedError
    
    def test_optimal_policies(
            self,
            num_episodes,
            epsilon,
            gamma):
        
        """
        Computes optimal policies for each reward machine and evaluates them 
            using epsilon-greedy exploration.

        params
        ----------
        num_episodes(int): Number of evaluation episodes
        epsilon(float): Epsilon constant for exploring the environment
        gamma(float): Discount factor

        returns
        -----------
        List with the optimal average-reward-per-step per reward machine
        """
        S, L, A, T = self.env.get_model()
        print("\nComputing optimal policies... ", end='', flush=True)
        optimal_policies = [value_iteration(S, A, L, T, rm, gamma) for rm in self.reward_machines]
        print("Done!")
        optimal_ARPS = [[] for _ in range(len(optimal_policies))]
        print("\nEvaluating optimal policies.")
        for ep in range(num_episodes):
            if ep % 100 == 0 and ep > 0:
                print("%d%d" % (ep, num_episodes))
            self.reset()
            s = tuple(self.obs)
            u = self.current_u_id
            rm_id = self.current_rm_id
            rewards = []
            done = False
            while not done:
                a = random.choice(A) if random.random() < epsilon else optimal_policies[rm_id][(s, u)]
                _, r, done, _ = self.step(a)
                rewards.append(r)
                s = tuple(self.obs)
                u = self.current_u_id
            optimal_ARPS[rm_id].append(sum(rewards) / len(rewards))
        print("DOne!\n")

        return [sum(arps) / len(arps) for arps in optimal_ARPS]
    

class OfficeRMEnv(GridRMEnv):
    
    def __init__(self):
        rm_files = [
            "./envs/grids/reward_machines/office/t%d.txt" % i for i in range(1, 5)
        ]
        env = OfficeWorld()
        super().__init__(GridEnv(env), rm_files)
    

