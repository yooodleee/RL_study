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
    Union)

import collections
import itertools
import copy
import numpy as np
import torch
import snappy


from . import types as types_lib


CompressedArray = Tuple[bytes, Tuple, np.dtype]


# Generic replay structure: Any flat named tuple.
ReplayStructure = TypeVar('ReplayStructure', bound=Tuple[Any, ...])


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
                        done = None)


class UniformReplay(Generic[ReplayStructure]):
    """
    Uniform replay, with circular buffer storage for flat named tuples.
    """

    def __init__(
            self,
            capacity: int,
            structure: ReplayStructure,
            random_state: np.random.RandomState,    
            time_major: bool = False,
            encoder: Optional[Callable[[ReplayStructure], Any]] = None,
            decoder: Optional[Callable[[Any], ReplayStructure]] = None):
        
        if capacity <= 0:
            raise ValueError(
                f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._storage = [None] * capacity
        self._num_added = 0

        self._time_major = time_major

        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)
    
    def add(self, item: ReplayStructure) -> None:
        """
        Adds single item to replay.
        """
        self._storage[self._num_added % self._capacity] = self._encoder(item)

        self._num_added += 1
    
    def get(self, indices: Sequence[int]) -> List[ReplayStructure]:
        """
        Retrieves items by indices.
        """
        return [self._decoder(self._storage[i]) for i in indices]
    
    def sample(self, batch_size: int) -> ReplayStructure:
        """
        Sample batch if items replay uniformly, with replacement.
        """
        if self.size < batch_size:
            raise RuntimeError(
                f'Replay only have {self.size} samples, got sample batch size 
                {batch_size}')
        
        indices = self._random_state.randint(self.size, size=batch_size)
        samples = self.get(indices)
        return np_stack_list_of_transitions(
            samples, self.structure, self.stack_dim)
    
    @property
    def stack_dim(self) -> int:
        """
        Stack dimension, for RNN we may need to make the tensor time major
        by stacking on second dimension [T, B, ...].
        """
        if self._time_major:
            return 1
        else:
            return 0
    
    @property
    def size(self) -> int:
        """
        Number of items currently contained in replay.
        """
        return min(self._num_added, self._capacity)
    
    @property
    def capacity(self) -> int:
        """
        Total capacity of replay (max number of items stored at any one time).
        """
        return self._capacity
    
    @property
    def num_added(self) -> int:
        """
        Total number of sample added to the replay.
        """
        return self._num_added
    
    def reset(self) -> None:
        """
        Reset the state of replay, should be called aat the beggining of every 
            episode
        """
        self._num_added = 0


class PrioritizedReplay:
    """
    Prioritized replay, with circular buffer storage for flat named tuples.
    This is the propotional variant as described in
        http://arxiv.org/abs/1511.05952.
    """

    def __init__(
            self,
            capacity: int,
            structure: ReplayStructure,
            priority_exponent: float,
            importance_sampling_exponent: float,
            random_state: np.random.RandomState,
            normalize_weights: bool = True,
            time_major: bool = False,
            encoder: Optional[Callable[[ReplayStructure], Any]] = None,
            decoder: Optional[Callable[[Any], ReplayStructure]] = None):
        
        if capacity <= 0:
            raise ValueError(
                f'Expect capacity to be a positive integer, got {capacity}')
        self.structure = structure
        self._capacity = capacity
        self._random_state = random_state
        self._encoder = encoder or (lambda s: s)
        self._decoder = decoder or (lambda s: s)

        self._storage = [None] * capacity
        self._num_added = 0

        self._time_major = time_major

        self._priorities = np.ones((capacity,), dtype=np.float32)
        self._priority_exponent = priority_exponent
        self._importance_sampling_exponent = importance_sampling_exponent

        self._normalize_weights = normalize_weights
    
    def add(self, item: ReplayStructure, priority: float) -> None:
        """
        Adds a single item with a given priority to the replay buffer.
        """
        if not np.isfinite(priority) or priority < 0.0:
            raise ValueError('priority must be finite and positive.')
        
        index = self._num_added % self._capacity
        self._priorities[index] = priority
        self._storage[index] = self._encoder(item)
        self._num_added += 1

    def get(self, indices: Sequence[int]) -> Iterable[ReplayStructure]:
        """
        retrieves transitions by indicies.
        """
        return [self._decoder(self._storage[i]) for i in indices]
    
    def sample(
            self, 
            size: int) -> Tuple[ReplayStructure, np.ndarray, np.ndarray]:
        """
        Samples a batch of transitions.
        """
        if self.size < size:
            raise RuntimeError(
                f'Replay only have {self.size} samples, got sample size 
                {size}')
        
        if self._priority_exponent == 0:
            indices = self._random_state.uniform(0, self.size, size=size)\
                .astype(np.float32)
            weights = np.ones_like(indices, dtype=np.float32)
        else:
            # code copied from seed_rl
            priorities = self._priorities[: self.size] \
                        ** self._priority_exponent
            
            probs = priorities / np.sum(priorities)
            indices = self._random_state.choice(
                np.arange(probs.shape[0]), size=size, replace=True, p=probs)

            # Importance weights.
            weights = ((1.0 / self.size) / np.take(probs, indices)) \
                    ** self._importance_sampling_exponent
            
            if self._normalize_weights:
                weights /= np.max(weights) + 1e-8   # Normalize.

        samples = self.get(indices)
        stacked = np_stack_list_of_transitions(
            samples, self.structure, self.stack_dim)
        return stacked, indices, weights
    
    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        """
        Updates indices with given priorities.
        """
        priorities = np.asarray(priorities)
        if not np.isfinite(priorities).all() \
            or (priorities < 0.0).any():
            raise ValueError('priorities must be finite and positive.')
        for index, priority in zip(indices, priorities):
            self._priorities[index] = priority
    
    @property
    def stack_dim(self) -> int:
        """
        Stack dimension, for RNN we may need to make the tensor time major 
        by stacking on second dimension as [T, B, ...].
        """
        if self._time_major:
            return 1
        else:
            return 0
    
    @property
    def size(self) -> None:
        """
        Number of elements currently contained in replay.
        """
        return min(self._num_added, self._capacity)
    
    @property
    def capacity(self) -> None:
        """
        Total capacity of replay (maximum number of items that can be stored.)
        """
        return self._capacity
    
    @property
    def importance_sampling_exponent(self):
        """
        Importance sampling exponent at current step.
        """
        return self._importance_sampling_exponent(self._num_added)


class GradientReplay(Generic[ReplayStructure]):
    """
    Store and retrieve aggregated network gradients for training A2C agent
    with gradients parallelism method.
    """

    def __init__(
            self,
            capacity: int,
            network: torch.nn.Module,
            compress: bool) -> None:
            
        if capacity <= 0:
            raise ValueError(
                f'Expect capacity to be a positive integer, got {capacity}')
        super().__init__()
        self._capacity = capacity
        self._decode = uncompress_array if compress else lambda s: s

        self._num_added = 0

        # Get number of layers in the network
        params = list(network.parameters())
        self._num_layers = len(params)
        del params

        # Create a list of lists (for each layer) to store gradients
        # with outer list size num_layers, inner list size maxsize
        self._gradients = [[None] 
                           * self._capacity for _ in range(self._num_layers)]
    
    def add(self, gradients: List[np.ndarray]) -> None:
        """
        Store extracted gradients with [param.grad.data.cpu().numpy()
        for param net.parameters()]
        """
        assert len(gradients) == self._num_layers

        for i, grad_layer_i in enumerate(gradients):    # for each layer
            j = self._num_added % self._capacity    # current batch index
            self._gradients[i][j] = self._decode(grad_layer_i)

        self._num_added += 1
    
    def sample(self)-> List[np.ndarray]:
        """
        Aggregate stored gradients by batch size and clear internal state
        """
        gradients = []

        for batch_grad_layer_i in self._gradients:
            grad_array = np.stack(batch_grad_layer_i, axis=0)\
                            .astype(np.float32)    # [batch_size, layer_shape]
            gradients.append(grad_array)
        
        self.reset()
        return gradients
    
    def reset(self) -> None:
        """
        Reset size counter is enough.
        """
        self._num_added = 0
    
    @property
    def num_layers(self) -> int:
        """
        Returns number of layers in the network.
        """
        return self._num_layers
    
    @property
    def size(self) -> int:
        """
        Returns added samples.
        """
        return self._num_added


class TransitionAccumulator:
    """
    Accumulates timesteps to form transitions.
    """

    def __init__(self):
        self._timestep_tm1 = None
        self._a_tm1 = None

    def step(self, timestep_t: types_lib.TimeStep, 
             a_t: int)-> Iterable[Transition]:
        """
        Accumulates timestep and resulting action, maybe yield a transition.

        We only need the s_t, r_t, and done flag for a given timestep_t
        the first timestep yield nothing since we don't have a full transition

        if the given timestep_t transition is terminal state, we need to reset 
            the state of the accumulator,
        so the next timestep which is the start of a new episode yields nothing
        """
        if timestep_t.first:
            self.reset()
        
        if self._timestep_tm1 is None:
            if not timestep_t.first:
                raise ValueError(
                    f'Expected first timestep, got {str(timestep_t)}')
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.

        transition = Transition(
                        s_tm1=self._timestep_tm1.observation,
                        a_tm1=self.a_tm1,
                        r_t=timestep_t.reward,
                        s_t=timestep_t.observation,
                        done=timestep_t.done)
        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t
        yield transition
    
    def reset(self) -> None:
        """
        Rests the accumulator.
        Following timestep is expected to be 'FIRST'.
        """
        self._timestep_tm1 = None
        self._a_tm1 = None


def _build_n_step_transition(
        transitions: Iterable[Transition],
        discount: float) -> Transition:
    """
    Builds a single n-step transition from n 1-step transitions.
    """
    r_t = 0.0
    discount_t = 1.0
    for transition in transitions:
        r_t += discount_t * transition.r_t
        discount_t *= discount
    
    return Transition(
        s_tm1 = transitions[0].s_tm1,
        a_tm1 = transitions[0].a_tm1,
        r_t = r_t,
        s_t = transitions[-1].s_t,
        done = transitions[-1].done,
    )


class NStepTransitionAccumulator:
    """
    Accumulates timesteps to form n-step transitions.

    Let `t` be the index of timestep within an episode and `T` be the index of
        the final timestep within an episode. Then given the step type of the 
        timestep passed into `step()` the accumulator will:
    *   `FIRST`: yield nothing.
    *   `MID`: if `t < n`, yield nothing, else yield one n-step transition
        `s_{t - n} -> s_t`.
    *   `LAST`: yield all transitions that end at `s_t = s_T` from up to n 
        steps away, specifically, `s_{T - min(n, T)} -> s_T, ..., s_{T - 1} 
        -> s_T`. These are `min(n, T)`-step, ..., `1`-step transitions.
    """

    def __init__(self, n, discount):
        self._discount = discount
        # Store 1-step transitions.
        self._transitions = collections.deque(maxlen=n) 
        self._timestep_tm1 = None
        self._a_tm1 = None

    def step(self, timestep_t: types_lib.TimeStep, 
             a_t: int)-> Iterable[Transition]:
        """
        Accumulates timestep and resulting action, yields transitions.
        """
        if timestep_t.first:
            self.reset()
        
        # There are no transitions on the first timestep.
        if self._timestep_tm1 is None:
            assert self._a_tm1 is None
            if not timestep_t.first:
                raise ValueError(
                    f'Expected first timestep, got {str(timestep_t)}')
            self._timestep_tm1 = timestep_t
            self._a_tm1 = a_t
            return  # Empty iterable.
        
        self._transitions.append(Transition(
                                    s_tm1 = self._timestep_tm1.observation,
                                    a_tm1 = self._a_tm1,
                                    r_t = timestep_t.reward,
                                    s_t = timestep_t.observation,
                                    done = timestep_t.done))

        self._timestep_tm1 = timestep_t
        self._a_tm1 = a_t

        if timestep_t.done:
            # Yield any remaining n, n-1, ..., 1-step transitions at episode 
            # end.
            while self._transitions:
                yield _build_n_step_transition(self._transitions, 
                                               self._discount)
                self._transitions.popleft()
        else:
            # Wait for n transitions before yielding anything.
            if len(self._transitions) < self._transitions.maxlen:
                return  # Empty iterable.
            
            assert len(self._transitions) == self._transitions.maxlen

            # This is the typical case, yield a single n-step transition.
            yield _build_n_step_transition(self._transitions, self._discount)
    
    def reset(self) -> None:
        """
        Resets the accumulator.
        Following timestep is expected to be FIRST.
        """
        self._transitions.clear()
        self._timestep_tm1 = None
        self._a_tm1 = None


class Unroll:
    """
    Unroll transitions to a specific timestep, used for RNN networks
    like R2D2, IMPALA, support cross episodes and do not cross episodes.
    """

    def __init__(
            self,
            unroll_length: int,
            overlap: int,
            structure: ReplayStructure,
            cross_episode: bool = True) -> None:
        """
        Args:
            unroll_weight: the unroll length
            overlap: adjacent unrolls overlap.
            structure: transition structure, used to stack sequence of unrolls
                into a single transition.
            cross_episode: should unroll cross episode, default on.
        """

        self.structure = structure
        self._unroll_length = unroll_length
        self._overlap = overlap
        self._full_unroll_length = unroll_length + overlap
        self._cross_episode = cross_episode

        self._storage = collections.deque(maxlen=self._full_unroll_length)

        # Persist last unrolled transitions incase not cross episode.
        # Sometimes the episode ends without reaching a full 'unroll length',
        # we will reuse some transition from last unroll to try to make a 
        # 'full length unroll'.
        self._last_unroll = None

    def add(self, transition: Any, 
            done: bool) -> Union[ReplayStructure, None]:
        """
        Add new transition into storage.
        """
        self._storage.append(transition)

        if self.full:
            return self._pack_unroll_into_single_transition()
        if done:
            return self._handle_episode_end()
        return None
    
    def _pack_unroll_into_single_transition(
            self)-> Union[ReplayStructure, None]:
        """
        Return a single transition object with transitions stacked with
        the unroll structure.
        """
        if not self.full:
            return None
        
        _sequence = list(self._storage)
        # Save for last use.
        self._last_unroll = copy.deepcopy(_sequence)
        self._storage.clear()

        # Handling adjacnet unroll sequences overlapping
        if self._overlap > 0:
            for transition in _sequence[-self._overlap :]:  # noqa: E203
                self._storage.append(transition)
        return self._stack_unroll(_sequence)
    
    def _handle_episode_end(self) -> Union[ReplayStructure, None]:
        """
        Handle episode end, incase no cross episodes, try to build a full
        unroll if last unroll is available.
        """
        if self._cross_episode:
            return None
        if self.size > 0 and self._last_unroll is not None:
            # Incase episode ends without reaching a full 'unroll length'
            # Use whatever we got from current unroll, fill in the missing 
            # ones from previous sequence
            _suffix = list(self._storage)
            _prefix_indices = self._full_unroll_length - len(_suffix)
            _prefix = self._last_unroll[-_prefix_indices]
            _sequence = list(itertools.chain(_prefix, _suffix))
            return self._stack_unroll(_sequence)
        else:
            return None

    def reset(self):
        """
        Reset unroll storage.
        """ 
        self._storage.clear()
        self._last_unroll = None
    
    def _stack_unroll(self, sequence):
        if len(sequence) != self._full_unroll_length:
            raise RuntimeError(
                f'Expect sequence length to be {self._full_unroll_length}, got
                  {len(sequence)}')
        return np_stack_list_of_transitions(sequence, self.structure)
    
    @property
    def size(self):
        """
        Return current unroll size.
        """
        return len(self._storage)
    
    @property
    def full(self):
        """
        return is unroll full.
        """
        return len(self._storage) == self._storage.maxlen


def stack_list_of_transitions(transitions, structure, dim=0):
    """
    Stack list of transition objects into one transition object with lists
    of tensors on a given dimension (default 0)
    """

    transposed = zip(*transitions)
    stacked = [torch.stack(xs, dim=dim) for xs in transposed]
    return type(structure)(*stacked)



def np_stack_list_of_transitions(transitions, structure, axis=0):
    """
    Stack list of transition objects into one transition object with lists 
    of tensors on a given dimension (default 0)
    """

    transposed = zip(*transitions)
    stacked = [np.stack(xs, axis=axis) for xs in transposed]
    return type(structure)(*stacked)



def split_structure(
        input_,
        structure,
        prefix_length: int,
        axis: int = 0) -> Tuple[ReplayStructure]:
    """
    Splits a structure of np.array along the axis, default 0.
    """

    # Compatibility check.
    if prefix_length > 0:
        for v in input_:
            if v.shape[axis] < prefix_length:
                raise ValueError(
                    f'Expect prefix_length to be less or equal to 
                    {v.shape[axis]}, got {prefix_length}')
    
    if prefix_length == 0:
        return (None, input_)
    else:
        split = [np.split(
                    xs,
                    # for torch.split() [prefix_length, xs.shape[axis] - 
                    # prefix_length],
                    [prefix_length, xs.shape[axis]],    
                    axis=axis) for xs in input_]

        _prefix = [pair[0] for pair in split]
        _suffix = [pair[1] for pair in split]

        return (type(structure)(*_prefix), type(structure)(*_suffix))
    


def compress_array(array: np.ndarray) -> CompressedArray:
    """
    Compresses a numpy array with snappy.
    """
    return snappy.compress(array), array.shape, array.dtype



def uncompress_array(compressed: CompressedArray) -> np.ndarray:
    """
    Uncompresses a numpy array with snappy given its shape and dtype.
    """
    compressed_array, shape, dtype = compressed
    byte_string = snappy.uncompress(compress_array)
    return np.frombuffer(byte_string, dtype=dtype).reshape(shape)