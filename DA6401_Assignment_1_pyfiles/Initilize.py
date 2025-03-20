import numpy as np
from Activation_grad import *

class Initilize:
    def __init__(self, layer_dimension, activation_function, y_train, method = "Xavier_U"):
        self.n = layer_dimension
        self.activation_fn = activation_function
        self.y = y_train
        self.Init_method = method

class InitializeWeights(Initilize):
    def __init__(self, ip_size, op_size, activation_function, batch_size, method):
        super().__init__([ip_size, op_size], activation_function, batch_size, method)  # Properly inherit Initilize attributes
        self.i_size = ip_size
        self.o_size = op_size
        self.Init_weights()


    def Init_weights(self):
        np.random.seed(0)

        if self.Init_method == "Xavier_N":
          np.random.seed(0)
          a = np.sqrt(1 / self.i_size)
          self.weight = np.random.randn(self.o_size,self.i_size)*a

        elif self.Init_method == "Xavier_U":
          np.random.seed(0)
          a = np.sqrt(6 / (self.o_size + self.i_size))
          self.weight = np.random.uniform((-a), a,( self.o_size,self.i_size))

        elif self.Init_method == "He_N":
          np.random.seed(0)
          a = np.sqrt(2 / self.i_size)
          self.weight = np.random.randn(self.o_size,self.i_size)*a

        elif self.Init_method == "He_U":
          np.random.seed(0)
          a = np.sqrt(6 / self.i_size)
          self.weight = np.random.uniform(-a, a, (self.o_size,self.i_size))

        elif self.Init_method == "Random":
          np.random.seed(0)
          self.weight = np.random.randn(self.o_size,self.i_size)*0.01
        else:
          raise ValueError(f"Unknown initialization method: {self.Init_method}")


        # Initialize biases and activations
        self.bias = np.zeros((self.o_size, 1))
        self.a = np.zeros((self.o_size, len(self.y[1])))
        self.h = np.zeros((self.o_size, len(self.y[1])))

        # Activation function and its derivative
        self.g = apply_activation(self.activation_fn, self.a).do_activation()
        self.d_g = apply_activation(self.activation_fn, self.a).do_activation_derivative()

        # Gradients
        self.d_a = np.zeros_like(self.a)
        self.d_h = np.zeros_like(self.h)
        self.d_w = np.zeros_like(self.weight)
        self.d_b = np.zeros_like(self.bias)
        self.Weight_updates = np.zeros_like(self.weight)
        self.bias_updates = np.zeros_like(self.bias)



class Weight_bias(Initilize):
    def __init__(self, layer_dimension, activation_function, y_train, method="Xavier_U"):
        super().__init__(layer_dimension, activation_function, y_train, method)
        self.network = []


    def Init_network(self):

        for i in range(1, len(self.n)):
          self.network.append(InitializeWeights( self.n[i-1], self.n[i], self.activation_fn[i-1], self.y, self.Init_method))

        return self.network
