#!/usr/bin/env python
from __future__ import print_function
import go_vncdriver # need low python version(do not install)
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
import sys
import gym
from envs import create_env
from worker import FastSaver
from model import LSTMPolicy
import utils
import distutils.version
use_tf12_api = distutils.version.LooseVersion(tf.version) >= distutils.version.LooseVersion('0.12.0')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def interface(args):
    """
    It only restores LSTMPolicy architecture, and does inference using that.
    """
    # get address of checkpoints
    indir = os.path.join(args.log_dir, 'train')
    outdir = os.path.join(args.log_dir, 'interface') if args.out_dir is None else args.out_dir
    with open(indir + '/checkpoint', 'r') as f:
        first_line = f.readline().strip()
    ckpt = first_line.split(' ')[-1].split('/')[-1][:-1]
    ckpt = ckpt.split('-')[-1]
    ckpt = indir + '/model.ckpt-' + ckpt

    # define environment
    if args.record:
        env = create_env(
            args.env_id,
            client_id='0',
            remotes=None,
            envWrap=args.envWrap,
            designHead=args.designHead,
            record=True,
            noop=args.noop,
            acRepeat=args.acRepeat,
            outdir=outdir)
    else:
        env = create_env(
            args.env_id,
            client_id='0',
            remotes=None,
            envWrap=args.envWrap,
            designHead=args.designHead,
            record=True,
            noop=args.noop,
            acRepeat=args.acRepeat)
    
    numaction = env.action_space.n

    with tf.device("/cpu:0"):
        # define policy network
        with tf.