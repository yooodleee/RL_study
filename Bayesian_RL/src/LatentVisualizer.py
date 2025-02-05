import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as tdist
import numpy as np


ACTION_SPACE_SIZE = 6

"""
import argparse
# config: utf-8

# Take length 50 snippets and record the cummulative return for each one.
# Then determine ground truth labels based on this.

# In[1]:

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
    
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5

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
    
        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint

        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint
        
        
        agent.load(model_path)
        episode_count = 30

        for i in range(episode_count):
            done = False
            traj = []
            actions = []
            gt_rewards = []
            r = 0

            ob = env.reset()
            steps = 0
            acc_reward = 0

            # os.mkdir('images/' + str(checkpoint))
            frameno = 0
            while True:
                action = agent.act(ob, r, done)
                ob, r, done, info = env.step(action)
                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0]  # get rid of first dimension ob.shape = (1, 84, 84, 4)
                traj.append(ob_processed)
                actions.append(actions[0])
                # save_image(
                #   torch.from_numpy(ob_processed).permute(2, 0, 1).reshape(4 * 84, 84),
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



def create_training_data(
        demonstrations,
        num_trajs,
        num_snippets,
        min_snippet_length,
        max_snippet_length):

    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    times = []
    actions = []
    num_demos = len(demonstrations)

    # add full trajs (for use on Enduro)
    "aaa""
    for n in range(num_trajs):
        ti = 0
        tj = 0

        # only add trajectories that are different returns
        while(ti == tj):
            # pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)

        # create random partial trajs by finding random start frame and random skip frame
        si = np.random.randint(6)
        sj = np.random.randint(6)
        step = np.random.randint(3, 7)

        traj_i = demonstrations[ti][si::step]   # slice(start, stop, step)
        traj_j = demonstrations[tj][sj::step]

        if ti > tj:
            label = 0
        else:
            label = 1

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
    "aaa""

    
    # fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0

        # only add trajectories that are different returns
        while (ti == tj):
            
            # pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        
        # create random snippets
        # find min length of both demos to ensure pick a demo no earlier than that
        # chosen in worse preferred demo
        
        min_length = min(len(demonstrations[ti][0]), len(demonstrations[tj][0]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        
        if ti > tj: # pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            #print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj][0]) - rand_length + 1)

        else:   # ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            # print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(
                tj_start,
                len(demonstrations[ti][0]) - rand_length + 1
            )

        # skip everyother framestack to reduce size
        traj_i = demonstrations[ti][0][ti_start : ti_start + rand_length : 1]
        traj_j = demonstrations[tj][0][tj_start : tj_start + rand_length : 1]
        
        # skip everyother framestack to reduce size
        traj_i_actions = demonstrations[ti][1][ti_start : ti_start + rand_length : 1]
        traj_j_actions = demonstrations[tj][1][tj_start : tj_start + rand_length : 1]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1

        len1 = len(traj_i)
        len2 = len(list(range(ti_start, ti_start + rand_length, 1)))

        if len1 != len2:
            print("------------LENGTH MISMATCH!--------------")

        training_obs.append((traj_i, traj_j))
        training_labels.append(label)
        times.append(
            list(
                range(ti_start, ti_start + rand_length, 1)
            ),
            list(
                range(tj_start, tj_start + rand_length, 1)
            ),
        )


    print("maximum traj length", max_traj_length)
    return training_obs, training_labels, times, actions


"""
# %%