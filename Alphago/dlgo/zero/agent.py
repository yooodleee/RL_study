import numpy as np
from keras import optimizers    # SGD

from ..agent import Agent


class Branch:
    def __init__(self, prior):
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0


