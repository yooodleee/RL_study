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
    policy,
    num_local_steps,
    summary_writer,
    render,
    predictor,
    envWrap,
    noReward,
):
    """
    The logic of the thread runner. In brief, it constantly keeps on running
    the policy, and as long as the rollout exceeds a certain length, the thread
    runner appends the policy to the queue.
    """
    last_state = env.reset()
    last_features = policy.get_initial_features()   # reset lasm memory
    length = 0
    rewards = 0
    values = 0
    if predictor is not None:
        ep_bonus = 0
        life_bonus = 0
    
    while True:
        terminal_end = False
        rollout = PartialRollout(predictor is not None)

        for _ in range(num_local_steps):
            # run policy
            fetched = policy.act(last_state, *last_features)
            action, value_, features = fetched[0], fetched[1], fetched[2:]

            # run environment: get action_index from sampled one-hot 'action'
            stepAct = action.argmax()
            state, reward, terminal, info = env.step(stepAct)
            if noReward:
                reward = 0.
            if render:
                env.render()
            
            curr_tuple = [last_state, action, reward, value_, terminal, last_features]
            if predictor is not None:
                bonus = predictor.pred_bonus(last_state, state, action)
                curr_tuple += [bonus, state]
                life_bonus += bonus
                ep_bonus += bonus
            
            # collect the experience
            rollout.add(*curr_tuple)
            rewards += reward
            length += 1
            values += value_[0]

            last_state = state
            last_features = features

            timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
            if timestep_limit is None: timestep_limit = env.spec.timestep_limit
            if terminal or length >= timestep_limit:
                # prints summary of each life if envWrap == Ture else each game
                if predictor is not None:
                    print("Episode finished. Sum of shaped rewards: %.2f. Length: %d. Bonus: %.4f." % (rewards, length, life_bonus))
                    life_bonus = 0
                else:
                    print("Bonus finished. Sum of shaped rewards: %.2f. Length: %d." % (rewards, length))
                if 'distance' in info: print('Mario Distance Covered:', info['distance'])
                length = 0
                rewards = 0
                terminal_end = True
                last_features = policy.get_initial_features()   # reset lstm memory
                # TODO: don't reset when gym timestep_limit increases, bootstrap -- doesn't matter for atari?
                # reset only if it hasn't already reseted
                if length >= timestep_limit or not env.metadata.get('semantics.autoreset'):
                    last_state = env.reset()
            
            if info:
                # summarize full game including all lives (even if envWrap=True)
                summary = tf.summary()
                for k, v in info.items():
                    summary.value.add(tag=k, simple_value=float(v))
                if terminal:
                    summary.value.add(tag='global/episode_value', simple_value=float(values))
                    values = 0
                    if predictor is not None:
                        summary.value.add(tag='global/episode_bonus', simple_value=float(ep_bonus))
                        ep_bonus = 0
                summary_writer.add_summary(summary, policy.global_step.eval())
                summary_writer.flush()
            
            if terminal_end:
                break

        if not terminal_end:
            rollout.r = policy.value(last_state, *last_features)
        
        # once we have enough experience, yield it, and have the TreadRunner place it on a queue
        yield rollout


class A3C(object):
    def __init__(
        self,
        env,
        task,
        visualise,
        unsupType,
        envWrap=False,
        designHead='universe',
        noReward=False,
    ):
        """
        An implementation of the A3C algorithm that is reasonably well-tuned for the VNC environments.
        Below, we will have a modest amount of complexity due to the way TensorFlow handles data parallelism.
        But overall, we'll define the model, specify its inputs, and describe how the policy gradients step
        should be computed.
        """
        self.task = task
        self.unsup = unsupType is not None
        self.envWrap = envWrap
        self.env = env

        predictor = None
        numaction = env.action_space.n
        worker_device = "/job:worker/task:{}/cpu:0".format(task)

        with tf.device(tf.train.replace_device_setter(1, worker_device=worker_device)):
            with tf.variable_creator_scope("global"):
                self.network = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                self.global_step = tf.get_variable("global_step", [], tf.int32, initializer=tf.constant_initializer(0, dtype=tf.int32), trainalbe=False)

                if self.unsup:
                    with tf.variable_creator_scope("predictor"):
                        if 'state' in unsupType:
                            self.ap_network = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.ap_network = StateActionPredictor(env.observation_space.shape, numaction, designHead)
        
        with tf.device(worker_device):
            with tf.variable_creator_scope("local"):
                self.local_network = pi = LSTMPolicy(env.observation_space.shape, numaction, designHead)
                pi.global_step = self.global_step
                if self.unsup:
                    with tf.variable_creator_scope("predictor"):
                        if 'state' in unsupType:
                            self.local_ap_network = predictor = StatePredictor(env.observation_space.shape, numaction, designHead, unsupType)
                        else:
                            self.local_ap_network = predictor = StateActionPredictor(env.observation_space.shape, numaction, designHead)

            # computing a3c loss: https://arxiv.org/abs/1506.02438
            self.ac = tf.placeholder(tf.float32, [None, numaction], name="ac")
            self.adv = tf.placeholer(tf.float32, [None], name="adv")
            self.r = tf.placeholder(tf.float32, [None], name="r")
            log_prob_tf = tf.nn.log_softmax(pi.logits)
            prob_tf = tf.nn.softmax(pi.logits)
            # 1) the "policy gradients" loss: its derivative is preciesly the policy gradient
            # notice that self.ac is a placeholder that is provided externally.
            # adv will contain the advantages, as calculated in process_rollout
            pi_loss = -tf.reduce_mean(tf.reduce_sum(log_prob_tf * self.ac, 1) * self.adv)   # Eq (19)
            # 2) loss of value function: 12_loss = (x-y)^2/2
            vf_loss = 0.5 * tf.reduce_mean(tf.square(pi.vf - self.r))   # Eq (28)
            # 3) entropy to ensure randomness
            entropy = -tf.reduce_mean(tf.reduce_sum(prob_tf * log_prob_tf, 1))
            # final a3c loss: lr of critic is half of actor
            self.loss = pi_loss + 0.5 * vf_loss - entropy * Constants['ENTROPY_BETA']