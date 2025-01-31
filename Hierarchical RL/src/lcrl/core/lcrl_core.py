import numpy as np
import random


class LCRL:
    """
    LCRL

    
    Attributes
    ------------
    MDP:
        an object with the following properties.
        (1) a "step(action)" function describing the dynamics.
        (2) a "reset()" function that resets the state to an initial state.
        (3) a "state_label(state)" function that maps states to labels.
        (4) current state of the MDP is "current_state".
        (5) action space is "action_space" and all actions are enabled in
            each state.
    LDBA:
        an object of ./src/automata/LDBA.py
        a limit-deterministic Buchi automation.
    discount_factor: float
        discount factor (default 0.9)
    learning_rate: float
        learning rate (default 0.9)
    epsilon: float
        tuning parameter for the epsilon-greedy exploration scheme (defualt 0.1)

    
    Methods
    ----------
    train_ql(number_of_episodes, iteration_threshold, Q_initial_value)
        employs Episodic Q-learning to synthesise an optimal policy over
        the product MDP
    train_nfq(number_of_episodes, iteration_threshold, nfq_replay_buffer_size, num_of_hidden_neurons)
        employs Episodict Neural Fitted Q-Iteration to synthesise an optimal policy over the product MDP
    train_ddpg(number_of_episodes, iteration_threshold, ddpg_replay_buffer_size, num_of_hidden_nuerons)
        employs Episodic Deep Deterministic Policy Gradient to synthesise an optimal policy ober the product MDP
    reward(automaton_state)
        shapes the reward function according to the automaton frontier set
        for more details refer to the tool paper
    action_space_augmentiation()
        augments the action space whenever an epsilon transition is expected
        for more details refer to the tool paper.

    """

    def __init__(
            self,
            LDBA=None,
            discount_factor=0.9,
            learning_rate=0.9,
            decaying_learning_rate=True,
            epsilon=0.15):
        
        if MDP is None:
            raise Exception("LCRL expects an MDP object as input")

        self.MDP = MDP
        if LDBA is None:
            raise Exception("LCRL expects an LDBA object as input")

        self.LDBA = LDBA
        self.epsilon_transition_exists = True if LDBA.epsilon_transitions.__len__() > 0 else False
        self.gamma = discount_factor
        self.alpha = learning_rate

        self.decay_lr = decaying_learning_rate
        if self.decay_lr:
            self.alpha_initial_value = learning_rate
            self.alpha_final_value = 0.1

        self.epsilon = epsilon
        self.path_length = []
        self.Q = {}
        self.replay_buffers = {}
        self.Q_initial_value = 0
        self.early_interruption = 0
        self.q_at_initial_state = []
        self.successes_in_test = 0
        # ###### testing area ####### #
        self.test = False
        # ########################### #
    