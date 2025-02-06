
import math
import numpy as np



class CartPole:
    """
    CartPole env modelled as an MDP


    Attributes:
    -----------------
        gravity: (float)
            gravitational acceleration.
        masscart: (float)
            mass of the cart
        masspole: (float)
            mass of the pole
        length: (float)
            length of the pole
        force_mag: (float)
            magnitude of the applied force


    Methods:
    ------------------
        reset():
            resets the MDP state
        step(action):
            changes the state of the MDP upon executing an action, where
            the action set is { right, up, left, down, stay }.
        state_label: (state)
            outputs the label of input state
    """

    def __init__(self):

        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1

        self.total_mass = (self.masspole + self.masscart)

        self.length = 0.5
        self.polemass_length = (self.masspole * self.length)

        self.force_mag = 10.0
        self.tau = 0.02
        self.kinematics_integrator = 'euler'
        self.theta_threshold_radians = 12 * 2 * math.pi / 260
        self.x_threshold = 2.4

        self.currenst_state = np.random.uniform(
            low=-0.05,
            high=0.05,
            size=(4,),
        )
        self.action_space = [-1, 1]
    

    