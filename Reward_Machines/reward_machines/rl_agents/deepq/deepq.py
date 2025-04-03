import os
import tempfile


import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np


import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds


from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput


from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func



class ActWrapper(object):

    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None
    

    @staticmethod
    def load_act(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_param)
        sess = tf.compat.v1.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)
            
            zipfile.ZipFile(arc_path, "r", zipfile.ZIP_DEFLATED).extractall(td)
            load_variables(os.path.join(td, "model"))
        
        return ActWrapper(act, act_params)
    

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)
    

    def step(self, observation, **kwargs):
        # DQN doesn't use RNNs so ignore states and masks
        kwargs.pop('S', None)
        kwargs.pop('M', None)
        return self._act([observation], **kwargs), None, None, None
    

    def save_act(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")
        
        with tempfile.TemporaryDirectory() as td:
            save_variables(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, "w") as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)
    

    def save(self, path):
        save_variables(path)



def load_act(path):
    """Load act function that was returned by learn function.
    
    Params
    ------------
        - path: (str)
            path to the act func pickle
    
    Returns
    ------------
        - act: (ActWrapper)
            function that takes a batch of observations
            and returns actions.
    """
    return ActWrapper.load_act(path)



def learn(
        env,
        network,
        seed=None,
        use_crm=False,
        use_rs=False,
        lr=5e-4,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=100,
        checkpoint_freq=10000,
        checkpoint_path=None,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        param_noise=False,
        callback=None,
        load_path=None,
        **network_kwargs,
):
    """Train a deepq model.
    
    
    Params
    ---------------
    env: (gym.Env)
        environment to train on
    network: (string or a function)
        neural network to use as a q function approximator. If string, has to be one of the names of
        registered models in baselines.common.models (mlp, cnn, conv_only). If a function, should 
        take an observation tensor and return a latent variable tensor, which will be mapped to the 
        Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: (int or None)
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is
        used.
    use_crm: (bool)
        use counterfactual experience to train the policy
    use_rs: (bool)
        use reward shaping
    lr: (float)
        learning rate for adam optimizer
    total_timesteps: (int)
        num of env steps to optimizer for
    buffer_size: (int)
        size of the replay buffer
    exploration_fraction: (float)
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: (float)
        final value of random act probability
    train_freq: (int)
        update the model every `train_freq` steps.
    batch_size: (int)
        size of a batch sampled from replay buffer for training
    print_freq: (int)
        how often to print out training prgress set to None to disable printing
    checkpoint_freq: (int)
        how often to save the model. This is so that the best version is resorted at the end of the training.
        If you do not wish to restore the best version at the end of the training set this variable to None.
    learning_starts: (int)
        how many steps of the model to collect transitions for before learning starts
    gamma: (float)
        discount factor
    target_network_update_freq: (int)
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: (True)
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: (float)
        alpha parameter for prioritized replay buffer
    prioritized_replay_t0: (float)
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: (int)
        number of iterations ober which beta will be annealed from initial value to 1.0.
        If set to None equals to total_timesteps.
    prioritized_replay_eps: (float)
        epsilon to add to the TD errors when updating priorities.
    param_noise: (bool)
        whether or not to use parameter space noise (https://arxiv.org/1706.01905)
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm. If callback returns true
        training stops.
    load_path: (str)
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.


    Returns
    ---------------
    act: (ActWrapper)
        Wrapper over act function. Adds ability to save it and load it. See header of 
        baselines/deepq/categorical.py for details on the act function.
    """

    # Adjusting hyper-parameters by considering the num of RM states for crm
    if use_crm:
        rm_states = env.get_num_rm_states()
        buffer_size = rm_states * buffer_size
        batch_size = rm_states * batch_size
    
    # Create all the functions necessary to train the model
    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func(network, **network_kwargs)

    # capture the shape outside the closure so that the env obj is not serialized
    # by cloudpickle when serializing make_obs_ph
    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)
    
    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
        param_noize=param_noise,
    )

    act_params = {
        "make_obs_ph": make_obs_ph,
        "q_func": q_func,
        "num_actions": env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    # Create the replay buffer
    if prioritized_replay:
        replay_buffer = PrioritizedReplayBuffer(buffer_size, 
                                                alpha=prioritized_replay_alpha)
        if prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = total_timesteps
        beta_schedule = LinearSchedule(
            prioritized_replay_beta_iters,
            initial_p=prioritized_replay_beta0,
            final_p=1.0,
        )
    else:
        replay_buffer = ReplayBuffer(buffer_size)
        beta_schedule = None
    
    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(
        schedule_timesteps=int(exploration_fraction * total_timesteps),
        initial_q=1.0,
        final_q=exploration_final_eps,
    )

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True

    with tempfile.TemporaryDirectory() as td:
        td = checkpoint_path or td

        model_file = os.path.join(td, "model")
        model_saved = False

        if tf.compat.v1.train.latest_checkpoint(td) is not None:
            load_variables(model_file)
            logger.log('Loaded model from {}'.format(model_file))
            model_saved = True
        
        elif load_path is not None:
            load_variables(load_path)
            logger.log('Loaded model from {}'.format(load_path))
        
        for t in range(total_timesteps):
            if callback is not None:
                if callback(locals(), globals()): break
            
            # Take act and update exploration to the newest val
            kwargs = {}
            if not param_noise:
                update_eps = exploration.value(t)
                update_param_noise_threshold = 0.
            else:
                update_eps = 0.
                # Compute the threshold such that the KL divergence between perturbed and non-perturbed
                # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
                # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et el., 2017
                # for detailed explanation.
                update_param_noise_threshold = -np.log(1. - exploration.value(t)
                                                       + exploration.value(t)
                                                       / float(env.action_space.n))
                kwargs["reset"] = reset
                kwargs["update_param_noise_threshold"] = update_param_noise_threshold
                kwargs["update_param_noise_scale"] = True
            action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]
            env_action = action
            reset = False
            new_obs, rew, done, info = env.step(env_action)

            # Store transition in the replay buffer.
            if use_crm:
                # Adding