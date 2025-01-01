"""
Replay components for training agents.
"""

from typing import (
    Any,
    NamedTuple,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import collections
import itertools
import copy
import numpy as np
import torch
import snappy


# pylint: disable=import-error
import types as types_lib


CompressedArray = Tuple[bytes, Tuple, np.dtype]


# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar(
    'ReplayStructure', bound=Tuple[Any, ...]
)


class Transition(NamedTuple):
    """
    A full transition for general use case.
    """

    s_tm1: Optional[np.array]
    a_tm1: Optional[int]
    r_t: Optional[float]
    s_t: Optional[np.ndarray]
    done: Optional[bool]


TransitionStructure = Transition(
    s_tm1 = None, 
    a_tm1 = None, 
    r_t = None, 
    done = None,
)


class UniformReplay(Generic[ReplayStructure]):
    """
    Uniform replay, with circular buffer storage for flat named tuples.
    """

    def __init__(
        self,
        capacity: int,
        structure: ReplayStructure,
        random_state: np.random.RandomState,    # pylint: disable=no-member
        time_major: bool = False,
        encoder: Optional[Callable[[ReplayStructure], Any]] = None,
        decoder: Optional[Callable[[Any], ReplayStructure]] = None,
    ):
        if capacity <= 0:
            raise ValueError(
                f'Expect capacity to be a positive integer, got {capacity}'
            )
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._storage = [None] * capacity
        self._num_added = 0

        self._time_major = time_major

        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)
    
    def add(self, item: ReplayStructure)-> None:
        """
        Adds single item to replay.
        """
        self._storage[
            self._num_added % self._capacity
        ] = self._encoder(item)

        self._num_added += 1
    
    def get(
        self, indices: Sequence[int]
    )-> List[ReplayStructure]:
        """
        Retrieves items by indices.
        """
        return [
            self._decoder(self._storage[i]) for i in indices
        ]
    
    def sample(
        self, batch_size: int
    )-> ReplayStructure:
        """
        Sample batch if items replay uniformly, with replacement.
        """
        if self.size < batch_size:
            raise RuntimeError(
                f'Replay only have {self.size} samples, got sample batch size {batch_size}'
            )
        
        indices = self._random_state.randint(self.size, size=batch_size)
        samples = self.get(indices)
        return np_stack_list_of_transitions(
            samples, self.structure, self.stack_dim
        )
    
    @property
    def stack_dim(self)-> int:
        """
        Stack dimension, for RNN we may need to make the tensor time major
        by stacking on second dimension [T, B, ...].
        """
        if self._time_major:
            return 1
        else:
            return 0
    
    @property
    def size(self)-> int:
        """
        Number of items currently contained in replay.
        """
        return min(self._num_added, self._capacity)
    
    @property
    def capacity(self)-> int:
        """
        Total capacity of replay (max number of items stored at any one time).
        """
        return self._capacity
    
    @property
    def num_added(self)-> int:
        """
        Total number of sample added to the replay.
        """
        return self._num_added
    
    def reset(self)-> None:
        """
        Reset the state of replay, should be called aat the beggining of every episode
        """
        self._num_added = 0


