import random

import numpy as np
import psutil
import torch

from utils.utils import get_env_wrapper


class ReplayBuffer:
    """
    Since stable baselines 3 doesn't currently support a smart replay buffer 
        (more concretely the "lazy frames" feature)
    i.e. allocating (10^6, 84, 84) (~7 GB) for Atari and extracting 4 frames 
        as needed, instead of (10^6, 4, 84, 84),
    here is an efficient implementation.

    Note:   inspired by Berkley's implementation: 
        https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3

    Further improvements:
        * Much more concise (and hopefully readable)
        * Reports error if you don't have enough RAM in advance to allocate 
            this buffer
        * Fixed a subtle buffer boundary bug (start index edge case)

    """
    def __init__(
            self, 
            max_buffer_size, 
            num_last_frames_to_fatch=4, 
            frame_shape=[1, 84, 84], 
            crash_if_no_mem=True):
        
        self.max_buffer_size = max_buffer_size
        self.frame_width = frame_shape[2]
        self.num_previous_frames_to_fatch = num_last_frames_to_fatch

        self.device=torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu")

        # Create main buffer contains - all types are chosen so as to optimize
        #  memory consumption
        self.frames = np.zeros([self.max_buffer_size] + frame_shape, 
                               dtype=np.uint8)
        self.actions = np.zeros([self.max_buffer_size, 1], 
                                dtype=np.uint8)
        # we need extra precision for rewards
        self.rewards = np.zeros([self.max_buffer_size, 1], 
                                dtype=np.float32)  
        self.dones = np.zeros([self.max_buffer_size, 1], 
                              dtype=np.uint8)

        # Numpy dones lazy execution so it can happen that only after a while 
        # the training starts hitting the RAM limit
        # than the page swapping kicks in which will slow-down the training 
        # significantly only after hours of training!
        # hance the _check_eough_ram function
        self._check_enough_ram(crash_if_no_mem)
    
    #
    # public API functions
    #

    def sotre_frame(self, frame):
        self.frames[self.current_free_slot_index] = frame

        # circular logic
        self.current_free_slot_index = (self.current_free_slot_index + 1) \
                                        % self.max_buffer_size  
        self.current_buffer_size = min(self.max_buffer_size, 
                                     self.current_buffer_size + 1)

        # need to store (action, reward, done) at this index
        return self.current_free_slot_index -1  
    
    def store_action_reward_done(self, index, action, reward, done):
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = done
    
    def fetch_random_states(self, batch_size):
        assert self._has_enough_data(batch_size), \
            "Can't fetch states from the replay buffer - not enough data."
        # Uniform random sampling without replacement. -1 because always need 
        # to fetch the current and the immediate
        random_unique_indices=random.sample(range(self.current_buffer_size -1),
                                             batch_size)

        # shape = (B, C, H, W)
        states = self._postprocess_state(
            np.concatenate([self._fetch_state(i) 
                            for i in random_unique_indices], 0))
        # shape = (B, C, H, W)
        next_states = self._postprocess_state(
                        np.concatenate([self._fetch_state(i+2) 
                            for i in random_unique_indices], 0))
        # Long is needed because actions are used for indexing of tensors 
        # (PyTorch constraint)
        actions = torch.from_numpy(self.actions[random_unique_indices]).\
                    to(self.device).long()
        rewards = torch.from_numpy(self.actions[random_unique_indices]).\
                    to(self.device)
        # Float is needed because we'll be multiplying Q values with done 
        # flags (1-done actually)
        dones = torch.from_numpy(self.dones[random_unique_indices]).\
                to(self.device).float()

        return states, actions, rewards, next_states, dones
    
    def fatch_last_state(self):
        # shape = (1, C, H, W) where C - number of past frames, 4 for Atari
        return self._postprocess_state(
                    self._fetch_state((self.current_free_slot_index - 1) \
                                        % self.max_buffer_size))
    
    def get_current_size(self):
        return self.current_buffer_size
    
    #
    # Helper functions
    #

    def _fetch_state(self, end_index):
        """
        We fetch end_index frame and ("num_last_frames_to_fetch" - 1) last 
            frames (4 in total in the case of Atari)
            in order to generate a state.

        Replay buffer has 2 edge cases that we need to handle:
            1) start_index related:
                * index is "too close"* to 0 and our circular buffer is still 
                    not full, thus we don't have enough frames
                * index is "too close" to the buffer boundary we could mix 
                    very old/new observations

            2) done flag is True - we don't won't to take observations before 
                that index since it belongs to a different
            life or episode.

        Notes:
            * "too close" is defined by 'num_last_frames_to_fetch' variable
            * terminology: state consists out of multiple observations 
                (frames in Atari case)

        """
        # start index is included, end index is excluded <=> [)
        end_index += 1
        start_index = end_index - self.num_previous_frames_to_fatch
        start_index = self._handle_start_index_edge_cases(start_index, 
                                                          end_index)

        num_of_missing_frames = self.num_previous_frames_to_fatch \
                                - (end_index - start_index)

        # start_index : end_index indexing won't work if start_index < 0
        if start_index < 0 or num_of_missing_frames > 0:    
            # If there are missing frames, because of the above handled 
            # edge-cases, fill them with black frames as per
            # original DeepMind Lua imp: 
        # https://github.com/deepmind/dqn/blob/master/dqn/TransitionTable.lua#171
            state=[np.zeros_like(self.frames[0]) 
                   for _ in range(num_of_missing_frames)]

            for index in range(start_index, end_index):
                state.append(self.frames[index % self.max_buffer_size])
            
            # shape = (C, H, W) -> (1, C, H, W) where C - number of past 
            # frames, 4 for Atari
            return np.concatenate(state, 0)[np.newaxis, :]
        else:
            # return from (C, 1, H, W) to (1, C, H, W) where C number of past
            # frames, 4 for Atari
            return self.frames[start_index : end_index].\
                reshape(-1, self.frame_height, self.frame_width)[np.newaxis, :]
    
    def _postprocess_state(self, state):
        # numpy -> tensor, move to device, unit8 -> float, [0, 255] -> [0, 1]
        return torch.from_numpy(state).to(self.device).float().div(255)
    
    def _handle_start_index_edge_cases(self, start_index, end_index):
        # Edge case 1:
        if not self._buffer_full() and start_index < 0:
            start_index=0
        
        # Edge case 2:
        # Handle the case where start index crosses the buffer head pointer - 
        # the data before and after the head pointer
        # belongs to completely different episodes
        if self._buffer_full():
            if 0 < (self.current_free_slot_index - start_index) \
                % self.max_buffer_size < self.num_previous_frames_to_fatch:
                start_index=self.current_free_slot_index
        
        # Edge case 3:
        # A done flag marks a boundary between different episodes or lives 
        # either way we shouldn't take frames
        # before or at the done flag into consideration
        for index in range(start_index, end_index - 1):
            if self.dones[index % self.max_buffer_size]:
                start_index=index + 1
        
        return start_index
    
    def _buffer_full(self):
        return self.current_buffer_size == self.max_buffer_size
    
    def _has_enough_data(self, batch_size):
        # e.g. if buffer size is 32 need at least 32 frames hence <
        return batch_size < self.current_buffer_size    
    
    def _check_enough_ram(self, crash_if_no_mem):
        # beautify memory output - helper function
        def to_GBs(memory_in_bytes):   
            return f'{memory_in_bytes / 2**30:.2f} GBs'
        
        available_memory = psutil.virtual_memory().available
        required_memory = self.frames.nbytes + self.actions.nbytes \
                            + self.rewards.nbytes
        print(f'required memory = {to_GBs(required_memory)} GBs, available 
              memory = {to_GBs(available_memory)} GB')

        if required_memory > available_memory:
            message=f"Not enough memory to store the complete replay 
                        buffer! \n" \
                    f"required: {to_GBs(required_memory)} > available: 
                        {to_GBs(available_memory)} \n" \
                    f"Page swapping will make your training super slow once 
                        hit your RAM limit." \
                    f"You can either modify replay_buffer_size argument or set 
                        crash_if_no_mem to False to ignore it."
            if crash_if_no_mem:
                raise Exception(message)
            else:
                print(message)


# Basic replay buffer testing
if __name__ == "__main__":
    size=500000
    num_of_collection_steps = 10000
    batch_size = 32

    # Steps 0: Ctreate replay buffer and the env
    replay_buffer = ReplayBuffer(size)

    # NoFrameskip - receive every frame from mthe env whereas the version 
    # without NoFrameskip would give every 4th frame
    # v4 - actions we send to env are ececuted, whereas v0 would ignore the 
    # last action we sent with 0.25 probability
    env_id = "PongNoFrameskip-v4"
    env = get_env_wrapper(env_id)

    # Step 1: Collect experience
    frame = env.reset()

    for i in range(num_of_collection_steps):
        random_action=env.action_space.sample()

        # For some reason for Pong gym returns more than 3 actions.
        print(f"Sampling action {random_action} \
              - {env.unwrappered.get_action_meanings()[random_action]}")

        frame, reward, done, info = env.step(random_action)

        index=replay_buffer.sotre_frame(frame)
        replay_buffer.store_action_reward_done(index, random_action, 
                                               reward, done)
        
        if done:
            env.reset()
    
    # Step 2: Fetch states from the buffer
    states, actions, rewards, \
        next_states, dones = replay_buffer.fetch_random_states(batch_size)

    print(states.shape, next_states.shape, 
            actions.shape, rewards.shape, dones.shape)