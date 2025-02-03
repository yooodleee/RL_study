import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # or any {'0', '1', '2'}
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np
import argparse
# config: utf-8

# Take length 50 snippets and record the cumulative return for each one.
# Then determine ground truth labels based on this.

import sys
import pickle
import gym
from gym import spaces
import time
import random
from torchvision.utils import save_image
from run_test import *
from baselines.common.trex_utils import preprocess
import os


def generate_novice_demos(
        env,
        env_name,
        agent,
        model_dir):
    
    checkpoint_min = 50
    checkpoint_max = 600
    checkpoint_step = 50
    checkpoints = []

    if env_name == "enduro":
        checkpoint_min = 3100
        checkpoint_max = 3650
    
    """
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    """
    
    for i in range(
        checkpoint_min,
        checkpoint_max + checkpoint_step,
        checkpoint_step
    ):
        if i < 10:
            checkpoints.append('0000' + str(i))
        
        elif i < 100:
            checkpoints.append('000' + str(i))
        
        elif i < 1000:
            checkpoints.append('00' + str(i))
        
        elif i < 10000:
            checkpoints.append('0' + str(i))
    
    print(checkpoints)


    demonstrations = []
    learning_returns = []
    learning_rewards = []

    for checkpoint in checkpoints:
        
        model_path = model_dir + "/models/" + env_name + "_25" + checkpoint
        
        # if env_name == "seaquest":
        #       model_path = model_dir + "/models/" + env_name + "_5" + checkpoint

        agent.load(model_path)
        episode_count = 5   # 30
        
        for i in range(episode_count):
            done = False
            traj = []
            actions = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0
            
            # oc.mkdir('images/' + str(checkpoint))
            frameno = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, info = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0]  # get rid of first dimension ob.shape = (1, 84, 84, 4)

                traj.append(ob_processed)
                actions.append(action[0])
                # save_image(
                #   torch.from_numpy(ob_processed).permute(2, 0, 1).reshape(4*84, 84),
                #   'images/' + str(checkpoint) + '/' + str(frameno) + '_action_' + str(action[0]) + '.png')
                frameno += 1

                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]

                if done:
                    print(
                        "checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward)
                    )
                    break
            
            print("traj length", len(traj))
            print("demo length", len(demonstrations))

            demonstrations.append([traj, actions])
            learning_returns.append(acc_reward)
            learning_rewards.append(gt_rewards)
    

    return demonstrations, learning_returns, learning_rewards


