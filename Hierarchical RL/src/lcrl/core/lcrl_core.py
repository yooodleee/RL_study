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
    
    """