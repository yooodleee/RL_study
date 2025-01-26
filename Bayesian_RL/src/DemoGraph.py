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