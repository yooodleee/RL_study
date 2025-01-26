import time
from rllab.algos.base import RLAlgoritm
import rllab.misc.logger as logger
import rllab.plotter as plotter
from sandbox.rocky.tf.policies.base import Poilcy
import tensorflow as tf
from samplers.batch_sampler import BatchSampler
from samplers.vectorized_sampler import VectorizedSampler
import numpy as np
import pickle


class BatchPolopt(RLAlgoritm):
    """
    Base class for batch sampling-based policy optimization methods.
    This includes various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    def __init__(
            self,
            env,
            policy,
            baseline,
            scope=None,
            n_itr=500,
            start_itr=0,
            batch_size=5000,
            max_path_length=500,
            discount=0.99,
            gae_lambda=1,
            plot=False,
            pause_for_plot=False,
            positive_adv=False,
            center_adv=True,
            store_paths=False,
            whole_paths=True,
            fixed_horizon=False,
            sampler_cls=None,
            sampler_args=None,
            force_batch_sampler=False,
            **kwargs):
        
        """
        Params
            env: Environment
            policy: Policy
            baseline: Baseline
            scope: Scope for identifying the algorithm. Must be specified if running
                multiple algorithms simultaneously, each using different environments
                and policies
            n_itr: Number of iterations
            start_itr: Starting iteration
            batch_size: Number of samples per iteration
            max_path_length: Maximum length of a single rollout
            discount: Discount
            gae_lambda: Lambda used for generalized advantage estimation
            plot: Plot evaluation run after each iteration
            pause_for_plot: Whether to pause befor contiuing when plotting
            center_adv: Whether to rescale the advantages so that they have mean 0 and started
                deviation 1.
            positive_adv: Whether to shift the advantages so that they are always positive.
                When used in conjunction with center_adv the advantages will be standardized 
                before shifting.
            store_paths: Whether to save all paths data to the snapshot.
        """
        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.plot = plot
        self.pause_for_plot = pause_for_plot
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.store_paths = store_paths
        self.whole_paths = whole_paths
        self.fixed_horizon = fixed_horizon
        self.kwargs = kwargs
        if 'reset_init_path' in self.kwargs:
            assert 'horizon' in self.kwargs
            with open(kwargs['reset_init_path'], 'rb') as f:
                self.reset_initial_states = pickle.load(f)
        if sampler_cls is None:
            if self.policy.vectorized and not force_batch_sampler:
                sampler_cls = VectorizedSampler
            else:
                sampler_cls = BatchSampler
        if sampler_args is None:
            sampler_args = dict()
            self.init_opt()
    
    def start_worker(self):
        self.sampler.start_worker()
        if self.plot:
            plotter.init_plot(self.env, self.policy)
    
    def shutdown_worker(self):
        self.sampler.shutdown_worker()
    
    def obtain_samples(self, itr, determ=False):
        return self.sampler.obtain_samples(itr, determ)
    
    def process_samples(self, itr, paths):
        return self.sampler.process_samples(itr, paths)
    
    def evaluate_fixed_init_trajectories(
            self,
            reset_initial_states,
            horizon):
        
        def f(x):
            if hasattr(self.env.wrapped_env, 'wrapped_env'):
                inner_env = self.env.wrapped_env.wrapped_env
                observation = inner_env.reset(x)
            else:
                self.env.reset()
                half = int(len(x) / 2)
                inner_env = self.env.wrapped_env.env.unwrapped
                inner_env.set_state(x[:half], x[half:])
                observation = inner_env._get_obs()
            episode_reward = 0.0
            episode_cost = 0.0
            for t in range(horizon):
                action = self.policy.get_action(observation)[1]['mean'][None]
                # clipping
                action = np.clip(action, *self.env.action_space.bounds)
                next_observation, reward, done, info = self.env.step(action[0])
                cost = inner_env.cost_np(
                    observation[None],
                    action,
                    next_observation[None],
                )
                # Update observation
                observation = next_observation
                # Update cost 
                episode_cost += cost
                # Update reward
                episode_reward += reward
            # assert episode_cost + episode_reward <1e-2
            return episode_cost
        
        # Run evaluation in parallel
        outs = np.array(list(map(f, reset_initial_states)))
        # Return avg_eps_reward and avg_eps_cost accordingly
        return np.mean(outs)
    
    def train(self):
        if 'initialized_path' in self.kwargs:
            import joblib
            from utils import get_session
            sess = get_session(interactive=True, mem_frac=0.1)
            data = joblib.load(self.kwargs['initialized_path'])
            self.policy = data['policy']
            self.env = data['env']
            self.baseline = data['baseline']
            self.init_opt()
            sess.run(
                tf.assign(self.policy._l_std_param.param,
                          np.zeros(self.env.action_space.shape[0]))
            )
            uninitialized_vars = []
            for var in tf.global_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            init_new_vars_op = tf.variables_initializer(uninitialized_vars)
            sess.run(init_new_vars_op)
            self.start_worker()
            start_time = time.time()
            avg_eps_costs = []
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Obtaining samples...")
                    paths = self.obtain_samples(itr)
                    logger.log("Logging diagnostics...")
                    samples_data = self.process_samples(itr, paths)
                    logger.log("Logging diagnostics...")
                    self.log_diagnostics(paths)
                    logger.log("Optimizing policy...")
                    self.optmize_policy(itr, samples_data)
                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr, samples_data)   # , **kwargs