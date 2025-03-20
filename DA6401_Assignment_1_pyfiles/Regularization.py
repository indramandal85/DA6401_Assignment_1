import numpy as np

class Regularisation:
    def __init__(self, network, weight=0):
        self.network = network
        self.weight = weight

class L2_regularisation(Regularisation):
    def Apply_L2(self):
        """Returns L2 regularization loss for the given network."""
        L = len(self.network)
        res = 0
        for j in range(L):
            if np.isnan(self.network[j].weight).any():
                print(f"Warning: NaN detected in network weights at layer {j}")
                return 0  # Prevents NaN propagation
            res += 0.5 * np.sum(self.network[j].weight ** 2)
        return res

    def Apply_L2_grad(self, weight):
        """Returns L2 regularization gradient for the given weight matrix/tensor."""
        return 2 * weight

class L1_regularisation(Regularisation):
    def Apply_L1(self):
        """Returns L1 regularization loss for the given network."""
        L = len(self.network)
        res = 0
        for j in range(L):
            if np.isnan(self.network[j].weight).any():
                print(f"Warning: NaN detected in network weights at layer {j}")
                return 0  # Prevents NaN propagation
            res += (1 / 2) * np.sum(np.abs(self.network[j].weight))
        return res

    def Apply_L1_grad(self, weight):
        """Returns L1 regularization gradient for the given weight matrix/tensor."""
        return np.sign(weight)

class ApplyReg(Regularisation):
    def __init__(self, reg_function, network, weight=0):
        self.reg_function = reg_function
        super().__init__(network, weight)

    def do_reg(self):
        if self.reg_function == 'L2':
            return L2_regularisation(self.network).Apply_L2()
        if self.reg_function == 'L1':
            return L1_regularisation(self.network).Apply_L1()
        if self.reg_function == 'L2_d':
            return L2_regularisation(self.network).Apply_L2_grad(self.weight)  # Explicitly pass weight
        if self.reg_function == 'L1_d':
            return L1_regularisation(self.network).Apply_L1_grad(self.weight)  # Explicitly pass weight
