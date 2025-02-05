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


class Net(nn.Module):

    def __init__(self, ENCODING_DIMS):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1)
        self.conv4 = nn.Conv2d(32, 16, 3, stride=1)

        # min(784, max(64, ENCODING_DIMS * 2))
        intermediate_dimension = 128

        self.fc1 = nn.Linear(784, intermediate_dimension)
        self.fc_mu = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc_var = nn.Linear(intermediate_dimension, ENCODING_DIMS)
        self.fc2 = nn.Linear(ENCODING_DIMS, 1)

        self.reconstruct1 = nn.Linear(ENCODING_DIMS, intermediate_dimension)
        self.reconstruct2 = nn.Linear(intermediate_dimension, 1568)
        
        self.reconstruct_conv1 = nn.ConvTranspose2d(2, 4, 3, stride=1)
        self.reconstruct_conv2 = nn.ConvTranspose2d(4, 16, 6, stride=1)
        self.reconstruct_conv3 = nn.ConvTranspose2d(16, 16, 7, stride=2)
        self.reconstruct_conv4 = nn.ConvTranspose2d(16, 4, 10, stride=1)

        self.temporal_difference1 = nn.Linear(ENCODING_DIMS * 2, 1, bias=False)#ENCODING_DIMS)
        # self.temporal_difference2 = nn.Linear(ENCODING_DIMS, 1)
        
        self.inverse_dynamics1 = nn.Linear(ENCODING_DIMS * 2, ACTION_SPACE_SIZE, bias=False)#ENCODING_DIMS)
        # self.inverse_dynamics2 = nn.Linear(ENCODING_DIMS, ACTION_SPACE_SIZE)

        self.forward_dynamics1 = nn.Linear(ENCODING_DIMS + ACTION_SPACE_SIZE, ENCODING_DIMS, bias=False)# (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        # self.forward_dynamics2 = nn.Linear((ENCODING_DIMS + ACTION_SPACE_SIZE) * 2, (ENCODING_DIMS + ACTION_SPACE_SIZE) * 2)
        # self.forward_dynamics3 = nn.Linear((ENCODIG_DIMS + ACTION_SPACE_SIZE) * 2, ENCODING_DIMS)

        self.normal = tdist.Normal(0, 1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        print(
            "Intermediate dimension calculated to be: " + str(intermediate_dimension)
        )


    def reparameterize(self, mu, var):
        """
        var is actually the log variance
        """

        if self.training:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            std = var.mul(0.5).exp()
            eps = self.normal.sample(mu.shape).to(device)

            return eps.mul(std).add(mu)
        else:
            return mu
    

    def cum_return(self, traj):
        # print("input shape of trajectory: ")
        # print(traj.shape)
        """
        calculate cumulative return of trajectory.
        """
        sum_rewards = 0
        sum_abs_rewards = 0
        
        x = traj.permute(0, 3, 1, 2)    # get into NCHW format
        # compute forward pass of reward network
        # parallelize across frames so batch size is length of partial trajecotry
        
        print("pre any: ", x.shape)
        x = F.leaky_relu(self.conv(x))
        print("after conv1: ", x.shape)

        x = F.leaky_relu(self.conv2(x))
        print("after conv2: ", x.shape)

        x = F.leaky_relu(self.conv3(x))
        print("after conv3: ", x.shape)

        x = F.leaky_relu(self.conv4(x))
        print("after conv4: ", x.shape)

        x = x.view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        
        z = self.reparameterize(mu, var)
        # print("after fc_mu: ", x.shape)

        r = self.fc2(z)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))

        return sum_rewards, sum_abs_rewards, mu, var, z
    

    def estimate_temporal_difference(self, z1, z2):
        x = self.temporal_difference1(torch.cat((z1, z2), 1))
        # x = self.temporal_difference2(x)

        return x
    

    def forward_dynamics(self, z1, actions):
        x = torch.cat((z1, actions), dim=1)
        x = self.forward_dynamics1(x)
        # x = F.leaky_relu(self.forward_dynamics2(x))
        # x = self.forward_dynamics3(x)

        return x
    

    def estimate_inverse_dynamics(self, z1, z2):
        concatenation = torch.cat((z1, z2), 1)
        x = self.inverse_dynamics1(concatenation)
        # x = F.leaky_relu(self.inverse_dynamics2(x))

        return x
    

    def decode(self, encodig):
        print("before: ", encodig.shape)
        x = F.leaky_relu(self.reconstruct1(encodig))
        print("after reconstruct1: ", x.shape)

        x = F.leaky_relu(self.reconstruct2(x))
        print("after reconstruct2: ", x.shape)

        x = x.view(-1, 2, 28, 28)
        # print("------decoding--------")
        # print(x.shape)

        x = F.leaky_relu(self.reconstruct_conv1(x))
        print("after reconstruct_conv1: ", x.shape)
        # print(x.shape)

        x = F.leaky_relu(self.reconstruct_conv2(x))
        print("after reconstruct_conv2: ", x.shape)
        # print(x.shape)
        # print(x.shape)

        x = F.leaky_relu(self.reconstruct_conv3(x))
        print("after reconstruct_conv3: ", x.shape)
        # print(x.shape)
        # print(x.shape)

        x = F.leaky_relu(self.reconstruct_conv4(x))
        print("after reconstruct_conv4: ", x.shape)
        # print(x.shape)
        # print("---------end decoding---------")

        return x.permute(0, 2, 3, 1)
    

    def forward(self, traj_i, traj_j):
        """
        Compute cumulative return for each traj and return logits.
        """
        cum_r_i, abs_r_i, mu1, var1, z1 = self.cum_return(traj_i)
        cum_r_j, abs_r_j, mu2, var2, z2 = self.cum_return(traj_j)

        return torch.cat(
            (
                cum_r_i.unsqueeze(0),
                cum_r_j.unsqueeze(0),
            ),
            0
        ), abs_r_i + abs_r_j, z1, z2, mu1, mu2, var1, var2
    


"""
def reconstruction_loss(decoded, target, mu, logvar):
    num_elements = decoded.numel()
    target_num_elements = decoded.numel()

    if num_elements != target_num_elements:
        print(
            "ELEMENT SIZE MISMATCH IN RECONSTRUCTION"
        )
        sys.exit()

    bce = F.binary_cross_entropy(decoded, target)
    kld = -.05 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld /= num_elements

    # print("bce: " + str(bce) + " kld: " + str(kld))

    return bce + kld


# =============== Train the network =====================

def learn_reward(
        reward_network,
        optimizer,
        training_inputs,
        training_outputs,
        training_times,
        training_actions,
        num_iter,
        l1_reg,
        checkpoint_dir):

    # check if gpu avialable
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    loss_criterion = nn.CrossEntropyLoss()
    temporal_diffrence_loss = nn.MSELoss()
    inverse_dynamics_loss = nn.CrossEntropyLoss()
    forward_dynamics_loss = nn.MSELoss()

    cum_loss = 0.0
    training_data = list(
        zip(
            training_inputs,
            training_outputs,
            training_times,
            training_actions,
        )
    )

    for epoch in range(num_iter):
        np.random.shuffle(training_data)

        training_obs, \
        training_labels, \
        training_times_sub, \
        training_actions_sub = zip(
            *training_data
        )
        validation_split = 0.9

        for i in range(len(training_labels)):
            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            times_i, times_j = training_times_sub[i]
            actions_i, actions_j = training_actions_sub[i]

            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)

            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            num_frames = len(traj_i)

            # zero out gradient
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, \
            abs_rewards, \
            z1, \
            z2, \
            mu1, \
            mu2, \
            logvar1, \
            logvar2, = reward_network.forward(
                traj_i, traj_j
            )
            outputs = outputs.unsqueeze(0)

            decoded1 = reward_network.decode(z1)
            # print("DECODE SHAPE: ")
            # print(decode1.shape)
            # print(decode1.type())
            # print("TRAJ_I SHAPE: ")
            # print(traj_i.shape)
            # print(traj_i.type())
            
            decoded2 = reward_network.decode(z2)
            reconstruction_loss_1 = 10 * reconstruction_loss(
                decoded1,
                traj_i,
                mu1,
                logvar1
            )
            reconstruction_loss_2 = 10 * reconstruction_loss(
                decoded2,
                traj_j,
                mu2,
                logvar2
            )

            t1_i = np.random.randint(0, len(times_i))
            t2_i = np.random.randint(0, len(times_i))
            t3_i = np.random.randint(0, len(times_j))
            t4_i = np.random.randint(0, len(times_j))

            est_dt_i = reward_network.estimate_temporal_difference(
                mu1[t1_i].unsqueeze(0),
                mu1[t2_i].unsqueeze(0)
            )
            est_dt_j = reward_network.estimate_temporal_difference(
                mu2[t2_j].unsqueeze(0),
                mu2[t2_j].unsqueeze(0)
            )
            real_dt_i = (times_i[t2_i] - times_i[t1_i]) / 100.0
            real_dt_j = (times_j[t2_j] - times_j[t2_j]) / 100.0

            actions_1 = reward_network.estimate_inverse_dynamics(
                mu1[0:-1], mu1[1:]
            )
            actions_2 = reward_network.estimate_inverse_dynamics(
                mu2[0:-1], mu2[1:]
            )
            target_actions_1 = torch.LongTensor(actions_i[1:]).to(device)
            target_actions_2 = torch.LongTensor(actions_j[1:]).to(device)
            
            # print((actions_1, target_actions_1))
            # print((actions_2, target_actions_2))

            inverse_dynamics_loss_1 = inverse_dynamics_loss(
                actions_1, target_actions_1
            ) / 1.9
            inverse_dynamics_loss_2 = inverse_dynamics_loss(
                actions_2, target_actions_2
            ) / 1.9

            # if epoch <= 1 else np.random.randint(1, min(1, max(epoch, 4)))
            forward_dynamics_distance = 5
            forward_dynamics_actions1 = target_actions_1
            forward_dynamics_actions2 = target_actions_2

            forward_dynamics_onehot_actions1 = torch.zeros(
                (
                    num_frames - 1,
                    ACTION_SPACE_SIZE
                ),
                dtype=torch.float32,
                device=device
            )
            forward_dynamics_onehot_actions2 = torch.zeros(
                (
                    num_frames - 1,
                    ACTION_SPACE_SIZE
                ),
                dtype=torch.float32,
                device=device
            )

            forward_dynamics_onehot_actions_1.scatter_(
                1, forward_dynamics_actions1.unsqueeze(1), 1.0
            )
            forward_dynamics_onehot_actions_2.scatter_(
                1, forward_dynamics_actions2.unsqueeze(1), 1.0
            )

            forward_dynamics_1 = reward_network.forward_dynamics(
                mu1[:-forward_dynamics_distance],
                forward_dynamics_onehot_actions_1[:(num_frames - forward_dynamics_distance)]
            )
            forward_dynamics_2 = reward_network.forward_dynamics(
                mu2[:-forward_dynamics_distance],
                forward_dynamics_onehot_actions_2[:(num_frames - forward_dynamics_distance)]
            )

            for fd in range(forward_dynamics_distance - 1):
                forward_dynamics_1 = reward_network.forward_dynamics(
                    forward_dynamics_1,
                    forward_dynamics_onehot_actions_1[fd_i + 1:(num_frames - forward_dynamics_distance + fd_i + 1)]
                )
                forward_dynamics_2 = reward_network.forward_dynamics(
                    forward_dynamics_2,
                    forward_dynamics_onehot_actions_2[fd_i + 1:(num_frames - forward_dynamics_distance + fd_i + 1)]
                )

            forward_dynamics_loss_1 = 100 * forward_dynamics_loss(
                forward_dynamics_1, mu1[forward_dynamics_distance:]
            )
            forward_dynamics_loss_2 = 100 * forward_dynamics_loss(
                forward_dynamics_2, mu2[forward_dynamics_distance:]
            )

            # print("est_dt: " + str(est_dt_i) + ", real_dt: " + str(real_dt_i))
            # print("est_dt: " + str(est_dt_j) + ", real_dt: " + str(real_dt_j))

            dt_loss_i = 4 * temporal_difference_loss(
                est_dt_i,
                torch.tensor(((real_dt_i,),), dtype=torch.float32, device=device)
            )
            dt_loss_j = 4 * temporal_difference_loss(
                est_dt_j,
                torch.tensor(((real_dt_j,),), dtype=torch.float32, device=device)
            )

            # l1_loss = 0.5 * (torch.norm(z1, 1) + torch.norm(z2, 1))
            # trex_loss = loss_criterion(outputs, labels)

            # loss = \
            #   trex_loss \
            #   + l1_reg \
            #   * abs_rewards \
            #   + reconstruction_loss_1 \
            #   + reconstruction_loss_2 \
            #   + dt_loss_i \
            #   + dt_loss_j \
            #   + inverse_dynamics_loss_1 \
            #   + inverse_dynamics_loss_2

            # reconstruction_loss_1 + reconstruction_loss_2 +
            loss = dt_loss_i \
                + dt_loss_j \
                + (inverse_dynamics_loss_1 + inverse_dynamics_loss_2) \
                + forward_dynamics_loss_1 \
                + forward_dynamics_loss_2 \
                + reconstruction_loss_1 \
                + reconstruction_loss_2 \
            
            if i < len(training_labels) * validation_split:
                print(
                    "TRAINING LOSS", end=" "
                )
            else:
                print(
                    "VALIDATION LOSS", end=" "
                )
            
            print(
                "dt_loss", 
                dt_loss_i.item(), 
                dt_loss_j.item(), 
                "inverse_dynamics", 
                inverse_dynamics_loss_1.item(), 
                inverse_dynamics_loss_2.item(), 
                "forward_dynamics", 
                forward_dynamics_loss_1.item(), 
                forward_dynamics_loss_2.item(), 
                "reconstruction", 
                reconstruction_loss_1.item(), 
                reconstruction_loss_2.item(), 
                end=" "
            )

            # loss = dt_loss_i \
                    + dt_loss_j \
                    + inverse_dynamics_loss_1 \
                    + inverse_dynamics_loss_2 \
                    + forward_dynamics_loss_1 \
                    + forward_dynamics_loss_2 \
                    + l1_loss
            # loss = forward_dynamics_loss_1 + forward_dynamics_loss_2
            # loss = inverse_dynamics_loss_1 + inverse_dynamics_loss_2
            # TODO and l2 reg

            # print(
            #   "!LOSSDATA " \
            #   + str(reconstruction_loss_1.data.numpy()) \
            #   + " " + str(reconstruction_loss_2.data.numpy()) \
            #   + " " + str(dt_loss_i.data.numpy()) \
            #   + " " + str(dt_loss_j.data.numpy()) \
            #   + " " + str(trex_loss.data.numpy()) \
            #   + " " + str(loss.data.numpy()) \
            #   + " " + str(inverse_dynamics_loss_1.data.numpy()) \
            #   + " " + str(inverse_dynamics_loss_2.data.numpy())
            # )

            # print(
            #   "!LOSSDATA " \
            #   + str(reconstruction_loss_1.data.numpy()) \
            #   + " " + str(reconstruction_loss_2.data.numpy()) \
            #   + " " + str(dt_loss_i.data.numpy()) \
            #   + " " + str(dt_loss_j.data.numpy()) \
            #   + " " + str(loss.data.numpy()) \
            #   + " " + str(inverse_dynamics_loss_1.data.numpy()) \
            #   + " " + str(inverse_dynamics_loss_2.data.numpy()) \
            #   + " " + str(forward_dynamics_loss_1.data.numpy()) \
            #   + " " + str(forward_dynamics_loss_2.data.numpy())
            # )

            # loss = inverse_dynamics_loss_1 + inverse_dynamics_loss_2
            # print(loss.data.numpy())
            # sys.stdout.flush()

            if i < len(training_labels) * validation_split:
                loss.backward()
                optimizer.step()

            # print stats to see if learning
            item_loss = loss.item()
            print("total", item_loss)
            cum_loss += item_loss

            if i % 100 == 99:
                # print(i)
                # print("epoch {}:{} loss {}".format(epoch, i, cum_loss))

                print(abs_rewards)
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)

    print("finished training")



def calc_accuracy(
        reward_network,
        training_inputs,
        training_outputs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0

    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]

            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            # forward to geet logits
            outputs, \
            abs_return, \
            z1, z2, _, _, _, _ = reward_network.forward(
                traj_i, traj_j
            )
            _, pred_label = torch.max(outputs, 0)

            if pred_label.item() == label:
                num_correct += 1.
    
    return num_correct / len(training_inputs)



def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []

    with torch.no_grad():
        for s in traj:
            r = net.cum_return(
                torch.from_numpy(np.array([s])).float().to(device)
            )[0].item()
            rewards_from_obs.append(r)

    return rewards_from_obs



def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))
"""


