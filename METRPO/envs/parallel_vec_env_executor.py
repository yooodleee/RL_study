import numpy as np
import pickle as pickle
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc import logger

from rllab.sampler.stateful_pool import singleton_pool
import uuid


def worker_init_envs(G, alloc, scope, env):
    logger.log(
        "initializing environment on worker %d" % G.worker_id
    )
    if not hasattr(G, 'parallel_vec_envs'):
        G.parallel_vec_envs = dict()
        G.parallel_vec_env_template = dict()
    G.parallel_vec_envs[scope] = [
        (idx, pickle.loads(pickle.dumps(env))) for idx in alloc
    ]
    G.parallel_vec_env_template[scope] = env


# For these two methods below, pack the data into batch numpy arrays whenever possible,
# to reduce communication cost
def worker_run_reset(G, flags, scope):
    if not hasattr(G, 'parallel_vec_envs'):
        logger.log("on worker %d" % G.worker_id)
        import traceback
        for line in traceback.format_stack():
            logger.log(line)
        # log the stacktrace at least
        logger.log("oops")
        for k, v in G.__dict__.items():
            logger.log(str(k) + " : " + str(v))
        assert hasattr(G, 'parallel_vec_envs')
    
    assert scope in G.parallel_vec_envs
    N = len(G.parallel_vec_envs[scope])
    env_template = G.parallel_vec_env_template[scope]
    obs_dim = env_template.observation_space.flat_dim
    ret_arr = np.zeros((N, obs_dim))
    ids = []
    flat_obs = []
    reset_ids = []
    for itr_idx, (idx, env) in enumerate(G.parallel_vec_envs[scope]):
        flag = flags[idx]
        if flag:
            flat_obs.append(env.reset())
            reset_ids.append(itr_idx)
        ids.append(idx)
    if len(reset_ids) > 0:
        ret_arr[reset_ids] = env_template.observation_space.flatten_n(flat_obs)
    return ids, ret_arr


def worker_run_step(G, action_n, scope):
    assert hasattr(G, 'parallel_vec_envs')
    assert scope in G.parallel_vec_envs
    env_template = G.parallel_vec_env_template[scope]
    ids = []
    step_results = []
    for (idx, env) in G.parallel_vec_envs[scope]:
        action = action_n[idx]
        ids.append(idx)
        step_results.append(tuple(env.step(action)))
    if len(step_results) == 0:
        return None
    obs, rewards, dones, env_infos = list(map(list, list(zip(*step_results))))
    obs = env_template.observation_space.flatten_n(obs)
    rewards = np.array(rewards)
    dones = np.array(dones)
    env_infos = tensor_utils.stack_tensor_dict_list(env_infos)

    return ids, obs, rewards, dones, env_infos


