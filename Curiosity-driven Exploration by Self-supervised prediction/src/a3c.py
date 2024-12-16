from __future__ import print_function
from collections import namedtuple
import numpy as np
import tensorflow as tf
from model import LSTMPolicy, StateActionPredictor, StatePredictor
import queue as queue
import scipy.signal
import threading
import distutils.version
from constants import Constants
use_tf12_api = distutils.version.LooseVersion(tf.VERSION) >= distutils.version.LooseVersion('0.12.0')

def disconut(x, gamma):
    """
        x = [r1, r2, r3, ..., rN]
        return [r1 + r2*gamma + r3*gamma + ...,
                    r2 + r3*gamma + r4*gamma + ...,
                        r3 + r4*gamma + r5*gamma + ...,
                            ..., ..., rN]
    """
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def process_rollout(rollout, gamma, lambda_=1.0, clip=False):
    """
    Given a rollout, compute its returns and the advantage.
    """
    # collecting transitions
    if rollout.unsup:
        batch_si = np.asarray(rollout.states + [rollout.end_state])
    else:
        batch_si = np.asarray(rollout.states)
    batch_a = np.asarray(rollout.actions)

    # collecting target for value network
    # V_t <-> r_t + gamma*r_{t+1} + ... + gamma^n*r_{t+n} + gamma^{n+1}*V_{n+1}
    rewards_plus_v = np.asarray(rollout.rewards + [rollout.r])  # bootstrapping
    if rollout.unsup:
        rewards_plus_v += np.asarray(rollout.bounses + [0])
    if clip:
        rewards_plus_v[:-1] = np.clip(rewards_plus_v[:-1], -Constants['REWARD_CLIP'], Constants['REWARD_CLIP'])
    batch_r = disconut(rewards_plus_v, gamma)[:-1]  # value network target

    # collecting target for policy network
    rewards = np.asarray(rollout.rewards)
    if rollout.unsup:
        rewards += np.asarray(rollout.bounses)
    if clip:
        rewards = np.clip(rewards, -Constants['REWARD_CLIP'], Constants['REWARD_CLIP'])
    vpred_t = np.asarray(rollout.values + [rollout.r])
    # "Generalized Advantage Estimation": https://arixv.org/abs/1506.02438
    # Eq (10): delta_t = Rt + gamma*V_{t+1} - V_t
    # Eq (16): batch_adv_t = delta_t + gamma*delta_{t+1} + gamma^2*delta_{t+2} + ...
    delta_t = rewards + gamma * vpred_t[1:] - vpred_t[:-1]
    batch_adv = disconut(delta_t, gamma * lambda_)

    features = rollout.features[0]

    return Batch(batch_si, batch_a, batch_adv, batch_r, rollout.terminal, features)


Batch = namedtuple("Batch", ["si", "a", "adv", "r", "terminal", "features"])


class PartialRollout(object):
    """
    A piece of a complete rollout. We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, unsup=False):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.r = 0.0
        self.terminal = False
        self.features = []
        self.unsup = unsup
        if self.unsup:
            self.bounses = []
            self.end_state = None
    
    def add(self, state, action, reward, value, terminal, features, bouns=None, end_state=None):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal
        self.features += [features]
        if self.unsup:
            self.bounses += [bouns]
            self.end_state = end_state
    
    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.r = other.r
        self.terminal = other.terminal
        self.features.extend(other.features)
        if self.unsup:
            self.bounses.extend(other.bounses)
            self.end_state = other.end_state


class RunnerThread(threading.Thread):
    """
    One of the key distinctions between a normal environment and a universe environment
    is that a universe environment is _real time_. This means that there should be a thread
    that would constantly interact with the environment and tell it what to do. This thread is here.
    """
    def __init__(self, env, policy, num_local_steps, visualise, predictor, envWrap, noReward):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(5) # ideally, should be 1. Mostly doesn't matter in this case.
        self.num_local_steps = num_local_steps
        self.env = env
        self.last_features = None
        self.policy = policy
        self.daemon = True
        self.sess = None
        self.summary_writer = None
        self.visualise = visualise
        self.predictor = predictor
        self.envWrap = envWrap
        self.noReward = noReward

    def start_runner(self, sess, summary_writer)-> None:
        self.sess = sess
        self.summary_writer = summary_writer
        self.start()
    
    def run(self)-> None:
        with self.sess.as_default():
            self._run()
    
    def _run(self)-> None:
        rollout_provider = env_runner(
            self.env,
            self.policy,
            self.num_local_steps,
            self.summary_writer,
            self.visualise,
            self.predictor,
            self.envWrap,
            self.noReward,
        )
        while True:
            # the timeout variable exists because apparently, if one worker dies, the other workers
            # won't die with it, unless the timeout is set to some large number. This is an empirical
            # observation.

            self.queue.put(next(rollout_provider), timeout=600.0)
    
def env_runner(
    env,
    policy
)