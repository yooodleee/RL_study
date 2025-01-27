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


