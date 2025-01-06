"""
Agent57 agent class.

From the paper "Agent57: Outperforming the Atari Human Benchmark.
    https://arxiv.org/pdf/2003.13350.
"""

from typing import (
    Iterable,
    Mapping,
    Optional,
    Tuple,
    NamedTuple,
    Text,
)
import copy
import multiprocessing
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import replay as replay_lib
from .. import types as types_lib
import normalizer
import transforms
import nonlinear_bellman
import base
import distributed
import bandit
from curiosity import EpisodicBounusModule, RndLifeLongBonusModule
from networks.value import Agent57NetworkInputs

torch.autograd.set_detect_anomaly(True)

HiddenState = Tuple[torch.Tensor, torch.Tensor]


class Agent57Transition(NamedTuple):
    """
    s_t, r_t, done are the tuple from env, step().

    last_action is the last agent took, before in s_t.
    """

    s_t: Optional[np.ndarray]
    a_t: Optional[int]
    q_t: Optional[np.ndarray]   # q values for s_t, computed from both ext_q_network and int_q_network
    prob_a_t: Optional[np.ndarray]  # probability of choose a_t in s_t
    last_action: Optional[int]  # for network input only
    ext_r_t: Optional[float]    # extrinsic reward for (s_tm1, a_tm1)
    int_r_t: Optional[float]    # intrinsic reward for (s_tm1)
    policy_index: Optional[int] # intrinsic reward scale beta index
    beta: Optional[float]   # intrinsic reward scale beta value
    discount: Optional[float]
    done: Optional[bool]
    ext_init_h: Optional[np.ndarray]    # nn.LSTM initial hidden state, from ext_q_network
    ext_init_c: Optional[np.ndarray]    # nn.LSTM initial cell state, from ext_q_network
    int_init_h: Optional[np.ndarray]    # nn.LSTM initial hidden state, from in_q_network
    int_init_c: Optional[np.ndarray]    # nn.LSTM initial cell state, from int_q_network


