from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import EnvSpec
from rllab.spaces.box import Box as TheanBox
from rllab.spaces.discrete import Discrete as TheanDiscrete
from rllab.spaces.product import Product as TheanProduct
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.spaces.box import Box
from sandbox.rocky.tf.spaces.product import Product
from cached_property import cached_property


def to_tf_space(space):
    if isinstance(space, TheanBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, TheanDiscrete):
        return Discrete(space.n)
    elif isinstance(space, TheanProduct):
        return Product(list(map(to_tf_space, space.components)))
    else:
        raise NotImplementedError


class WrappedCls(object):
    def __init__(
            self,
            cls,
            env_cls,
            extra_kwargs):
        
        self.cls = cls
        self.env_cls = env_cls
        self.extra_kwargs = extra_kwargs
    
    def __call__(self, *args, **kwargs):
        return self.cls(
            self.env_cls(*args, **dict(self.extra_kwargs, **kwargs))
        )


