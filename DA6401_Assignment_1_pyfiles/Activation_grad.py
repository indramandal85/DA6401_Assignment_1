import numpy as np

class CalActivation:
    def __init__(self, m):
        self.m = m

# Sigmoid (for hidden layers)
class sigmoid(CalActivation):
    def use_sigmoid(self):
        return 1 / (1 + np.exp(-np.clip(self.m, -250, 250)))  # Clip inputs

    def sigmoid_d(self):
        sig = self.use_sigmoid()
        return sig * (1 - sig)


class relu(CalActivation):
    def use_relu(self):
        return np.maximum(0, self.m)

    def relu_d(self):
        return np.where(self.m > 0, 1, 0)

class Leakyrelu(CalActivation):
    def use_relu(self):
        return np.maximum(0, self.m)

    def relu_d(self):
        return np.where(self.m > 0, 1, 0.01)


class softmax(CalActivation):
    def use_softmax(self):
        shifted = self.m - np.max(self.m, axis=0, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def softmax_d(self):
        z=self.m - np.max(self.m,axis=0)
        soft=np.exp(z)/np.sum(np.exp(z),axis=0)
        return soft*(1-soft)

class tanh(CalActivation):
    def use_tanh(self):
        return np.tanh(self.m)  # Built-in is stable

    def tanh_d(self):
        return 1 - (self.use_tanh() ** 2)



# This function call different activation function
class apply_activation(CalActivation):

    def __init__(self, activation_function, m):
        super().__init__(m)
        self.activation_function = activation_function.lower()

    def do_activation(self):
        if self.activation_function == 'sigmoid':
            return sigmoid(self.m).use_sigmoid()
        elif self.activation_function == 'relu':
            return relu(self.m).use_relu()
        elif self.activation_function == 'lrelu':
            return Leakyrelu(self.m).use_relu()
        elif self.activation_function == 'tanh':
            return tanh(self.m).use_tanh()
        elif self.activation_function == 'softmax':
            return softmax(self.m).use_softmax()

    def do_activation_derivative(self):
        if self.activation_function == 'sigmoid':
            return sigmoid(self.m).sigmoid_d()
        elif self.activation_function == 'relu':
            return relu(self.m).relu_d()
        elif self.activation_function == 'lrelu':
            return Leakyrelu(self.m).relu_d()
        elif self.activation_function == 'tanh':
            return tanh(self.m).tanh_d()
        elif self.activation_function == 'softmax':
            return softmax(self.m).softmax_d()
        else:
           raise ValueError("Unknown activation function")
