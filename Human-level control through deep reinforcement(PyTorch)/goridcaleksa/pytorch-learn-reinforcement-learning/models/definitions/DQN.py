from torch import nn
import torch
import numpy as np

from utils.utils import get_env_wrapper


class DQN(nn.Module):
    """
    I wrote the architecture a bit more generic, hence more lines of code,
    but it's more flaxible if you want to experiment with the DQN architecture.
    
    """
    def __init__(self, env, num_in_chennels=4, number_of_actions=3, epsilon_shcedule=None):
        super().__init__()
        self.env=env
        self.epsilon_shcedule=epsilon_shcedule  # defines the annealing strategy for epsilon in epsilon-greedy
        self.num_in_chennels=num_in_chennels    # counts the number of calls to epsilon function

        #
        # CNN params - from the Nature DQN paper - MODIFY this part if you want to experiment
        #
        num_of_filters_cnn=[num_in_chennels, 32, 64, 64]
        kernal_sizes=[8, 4, 3]
        strides=[4, 2, 1]

        #
        # Build CNN part of DQN
        #
        cnn_modules=[]
        for i in range(len(num_of_filters_cnn) - 1):
            cnn_modules.extend(
                self._cnn_block(num_of_filters_cnn, num_of_filters_cnn[i + 1], kernal_sizes[i], strides[i])
            )
        
        self.cnn_part=nn.Sequential(
            *cnn_modules,
            nn.Flatten()    # flatten from (B, C, H, W) into (B, C*H*W), where B is batch size and c number of in channels
        )

        # 
        # Build fully-connected part of DQN
        #
        with torch.no_grad():   # automatically figure out the shape for the given env observation
            # shape = (1, C', H, W), unit8, where C' is originally 1, i.e. grayscale frames
            dummy_input=torch.from_numpy(env.observation_space.sample()(np.newaxis))

            if dummy_input.shape[1] != num_in_chennels:
                assert num_in_chennels % dummy_input.shape[1] == 0
                # shape = (1, C, H, W), float
                dummy_input=dummy_input.repeat(1, int(num_in_chennels / dummy_input.shape[1]), 1, 1).float()
            
            num_nodes_fc1=self.cnn_part(dummy_input.shape[1])   # cnn output shape = (B, C*H*W)
            print(f"DQN's first FC layer input dimension: ", {num_nodes_fc1})
        
        #
        # FC params - MODIFY this part if you want to experiment
        #
        num_of_neurons_fc=[num_nodes_fc1, 512, number_of_actions]

        fc_modules=[]
        for i in range(len(num_of_neurons_fc) -1):
            last_layer= i == len(num_of_neurons_fc) -1  # last layer shouldn't have activation (Q-value is unbounded)
            fc_modules.extend(self._fc_block(num_of_neurons_fc[i], num_of_neurons_fc[i + 1], use_relu=not last_layer))
        
        self.fc_part=nn.Sequential(
            *fc_modules
        )
    
    def forward(self, states):
        # shape: (B, C, H, W) -> (B, NA) - where NA is the number of Actions
        return self.fc_part(self.cnn_part(states))
    
    def epsilon_greedy(self, state):
        assert self.epsilon_shcedule is not None, f"No schedule provided, can't call epsilon_greedy funtion"
        assert state.shape[0] == 1, f"Agent can only act on a single state"
        self.num_calls_to_epsilon_greedy += 1

        # Epsilon-greedy exploration
        if np.random.rand() < self.epsilon_value():
            # With epsilon probability act random
            action=self.env.action_space.sample()
        else:
            # Otherwise act greedly - choosing an action that maximizes Q
            # Shape evolution: (1, C, H, W) -> (forward) (1, NA) -> (argmax) (1, 1) -> [0] scalar
            action=self.forward(state).argmax(dim=1)[0].cpu().numpy()
        
        return action
    
    def epsilon_value(self):
        return self.epsilon_shcedule(self.num_calls_to_epsilon_greedy)
    
    # 
    # Helper/"private" functions
    #

    # The original CNN didn't use any padding: https://github.com/deepmind/dqn/blob/master/dqn/convnet.lua
    # not that it matters - it would probably work either way feel free to experiment with the architecture.
    def _cnn_block(self, num_in_filters, num_out_filters, kernal_size, stride):
        layers=[nn.Conv2d(num_in_filters, num_out_filters, kernel_size=kernal_size, stride=stride), nn.ReLU()]
        return layers
    
    def _fc_block(self, num_in_neurons, num_out_neurons, use_relu=True):
        layers=[nn.Linear(num_in_neurons, num_out_neurons)]
        if use_relu:
            layers.append(nn.ReLU())
        return layers


# Test DQN network
if __name__ == "__main__":
    # NoFrameskip - receive every frame from the env whereas the version without NoFrameskip would give every 4th frame
    # v4 - actions we send to env are excuted, whereas v0 would ignore the last action we sent with 0.25 probability
    env_id="PongNoFrameskip-v4"
    env_wrapped=get_env_wrapper(env_id)
    dqn=DQN(env_wrapped)    # testing only the __init__ function (mainly the automatic shape calculation mechanism)