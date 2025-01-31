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

    def train_ql(
            self,
            number_of_episodes,
            iteration_threshold,
            Q_initial_value=0):
        
        self.MDP.reset()
        self.LDBA.reset()
        self.Q_initial_value = Q_initial_value
    
        if self.LDBA.accepting_sets is None:
            raise Exception(
                "LDBA object is not defined properly. Please specify the 'accepting_set'."
            )
        
        # product MDP: ssynchronise the MDP state with the automation state
        current_state = self.MDP.current_state + [self.LDBA.automaton_state]
        product_MDP_action_space = self.MDP.action_space
        epsilon_transition_take = False

        # check for epsilon-transitions at the current automaton state
        if self.epsilon_transition_exists:
            product_MDP_action_space = self.action_space_augmentation()
        
        # initialize Q-value outside the main loop
        self.Q[str(current_state)] = {}
        for action_index in range(len(product_MDP_action_space)):
            self.Q[str(current_state)][product_MDP_action_space[action_index]] = Q_initial_value
        
        # main loop
        try:
            episode = 0
            self.path_length = [float("inf")]
            while episode < number_of_episodes:
                episode += 1
                self.MDP.reset()
                self.LDBA.reset()
                current_state = self.MDP.current_state + [self.LDBA.automaton_state]

                # check for epsilon-transitions at the current automaton state
                if self.epsilon_transition_exists:
                    product_MDP_action_space = self.action_space_augmentation()
                
                # Q value at the initial state
                Q_at_initial_state = []
                for action_index in range(len(product_MDP_action_space)):
                    Q_at_initial_state.append(
                        self.Q[str(current_state)][product_MDP_action_space[action_index]]
                    )
                # value function at the initial state
                self.q_at_initial_state.append(max(Q_at_initial_state))
                print(
                    'episode: ' + str(episode) 
                    + ', value function at s_0= ' + str(max(Q_at_initial_state))
                    # + ', trace length= ' + str(self.path_length[len(self.path_length) - 1])
                    # + ', learning_rate= ' + str(self.alpha)
                    # + ', current state= ' + str(self.MDP.current_state)
                )
                iteration = 0
                path = current_state

                # annealing the learning rate
                if self.decay_lr:
                    self.alpha = max(
                        self.alpha_final_value,
                        (
                            (self.alpha_final_value - self.alpha_initial_value) / (0.8 * number_of_episodes)
                        )
                        * episode + self.alpha_initial_value
                    )
                
                # each episode loop
                while self.LDBA.accepting_frontier_set and \
                        iteration < iteration_threshold and \
                        self.LDBA.automaton_state != -1:
                    iteration += 1

                    # find the action with max Q at the current state
                    Qs = []
                    for action_index in range(len(product_MDP_action_space)):
                        Qs.append(
                            self.Q[str(current_state)][product_MDP_action_space[action_index]]
                        )
                    maxQ_action_index = random.choice(np.where(Qs == np.max(Qs))[0])
                    maxQ_action = product_MDP_action_space[maxQ_action_index]

                    # check if an epsilon-transition is taken
                    if self.epsilon_transition_exists and \
                            maxQ_action_index > len(self.MDP.action_space) - 1:
                        epsilon_transition_take = True
                    
                    # product MDP modification (for more details refer to the tool paper)
                    if epsilon_transition_take:
                        next_MDP_state = self.MDP.current_state
                        next_automaton_state = self.LDBA.step(maxQ_action)
                    else:
                        # epsilon-greedy policy
                        if random.random() < self.epsilon:
                            next_MDP_state = self.MDP.step(random.choice(self.MDO.action_space))
                        else:
                            next_MDP_state = self.MDP.step(maxQ_action)
                        next_automaton_state = self.LDBA.step(self.MDP.state_label(next_MDP_state))
                    
                    # product MDP: synchronise the automaton with MDP
                    next_state = next_MDP_state + [next_automaton_state]

                    # check for epsilon-transitions at the next automaton state
                    if self.epsilon_transition_exists:
                        product_MDP_action_space = self.action_space_augmentation()
                    
                    # Q values of the next state
                    Qs_prime = []
                    if str(next_state) not in self.Q.keys():
                        self.Q[str(next_state)] = {}
                        for action_index in range(len(product_MDP_action_space)):
                            self.Q[str(next_state)][product_MDP_action_space[action_index]] = Q_initial_value
                            Qs_prime.append(Q_initial_value)
                    else:
                        for action_index in range(len(product_MDP_action_space)):
                            Qs_prime.append(
                                self.Q[str(next_state)][product_MDP_action_space[action_index]]
                            )
                    
                    # update the accepting frontier set
                    if not epsilon_transition_take:
                        reward_flag = self.LDBA.accepting_frontier_function(next_automaton_state)
                    else:
                        reward_flag = 0
                        epsilon_transition_take = False
                    
                    if reward_flag > 0:
                        state_dep_gamma = self.gamma
                    else:
                        state_dep_gamma = 1
                    
                    # Q update
                    self.Q[str(current_state)][maxQ_action] = \
                        (1 - self.alpha) * self.Q[str(current_state)][maxQ_action] + \
                        self.alpha * (self.reward(reward_flag) + state_dep_gamma * np.max(Qs_prime))
                    
                    if self.test:
                        print(
                            str(maxQ_action)
                            + ' | ' + str(next_state)
                            + ' | ' + self.MDP.state_label(next_MDP_state)
                            + ' | ' + str(reward_flag)
                            + ' | ' + str(self.Q[str(current_state)][maxQ_action])
                        )
                    
                    # update the state
                    current_state = next_state.copy()
                    path.append(current_state)
                
                # append the path length
                self.path_length.append(len(path))
        
        except KeyboardInterrupt:
            print('\nTraining exited early.')
            try:
                is_save = input(
                    'Wolud you like to save the training data? '
                    'If so, type in "y", otherwise, interrupt with CTRL+C. '
                )
            except KeyboardInterrupt:
                print('\nExiting...')
            
            if is_save == 'y' or is_save == 'Y':
                print('Saving...')
                self.early_interruption = 1
    
    