import numpy as np


class ExperienceCollector:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimate_values = []
    
    def begin_episode(self):
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimate_values = []
    
    def record_decision(
            self,
            state,
            action,
            estimated_value=0):
        
        self._current_episode_states.append(state)
        self._current_episode_actions.append(action)
        self._current_episode_estimate_values.append(estimated_value)
    
    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)
        self.states += self._current_episode_states
        self.actions += self._current_episode_actions
        self.rewards += [reward for _ in range(num_states)]

        for i in range(num_states):
            advantage = reward - \
                self._current_episode_estimate_values[i]
            self.advantages.append(advantage)
        
        self._current_episode_states = []
        self._current_episode_actions = []
        self._current_episode_estimate_values = []
    
    def to_buffer(self):
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            advantages=np.array(self.advantages),
        )


class ExperienceBuffer:
    def __init__(
            self,
            states,
            actions,
            rewards,
            advantages):
        
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.advantages = advantages
    
    def serialize(self, h5file):
        h5file.create_group('experience')
        h5file['experience'].create_dataset('states', data=self.states)
        h5file['experience'].create_dataset('actions', data=self.actions)
        h5file['experience'].create_dataset('rewards', data=self.rewards)
        h5file['experience'].create_dataset('advantages', data=self.advantages)
    

