import numpy as np


class Layer:
    def __init__(self):
        self.params = []

        self.previous = None
        self.next = None

        # state for the forward pass
        self.input_data = None
        self.output_data = None
        
        # state for the backward pass
        self.input_delta = None
        self.output_delta = None
    
    def connect(self, layer):
        self.previous = layer
        self.next = self
    
    def forward(self):
        raise NotImplementedError
    
    def get_forward_input(self):
        if self.previous is not None:
            return self.previous.output_data
        else:
            return self.input_data
    
    def backward(self):
        raise NotImplementedError
    
    def get_backward_input(self):
        if self.next is not None:
            return self.next.output_delta
        else:
            return self.input_delta
    
    def clear_deltas(self):
        pass

    def update_params(self, learning_rate):
        pass

    def describe(self):
        raise NotImplementedError


class ActivationLayer(Layer):
    def __init__(self, input_dim):
        super(ActivationLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = input_dim
    
    def forward(self):
        data = self.get_forward_input()
        self.output_data = sigmoid(data)
    
    def backward(self):
        delta = self.get_backward_input()
        data = self.get_forward_input()
        self.output_delta = delta * sigmoid_prime(data)
    
    def describe(self):
        print("|-- " + self.__class__.__name__)
        print(" |-- dimensions: ({}, {})".format(self.input_dim, self.output_dim))


