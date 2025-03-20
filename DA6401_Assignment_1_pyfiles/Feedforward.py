import numpy as np
from Activation_grad import *

class Pre_Feedforward:

    def __init__(self, inputs, w, b, activation_fn):
        self.ip= inputs
        self.w = w
        self.b = b
        self.activation_fn = activation_fn
        self.Preactivation_cal()


    def Preactivation_cal(self):

        A = np.dot(self.w, self.ip) + self.b
        A_norm = (A - np.mean(A, axis=1, keepdims=True)) / np.std(A, axis=1, keepdims=True) #Activation Normalization
        H = apply_activation(self.activation_fn, A_norm).do_activation()
        #cache = (H, A)
        #self.cache["Input"] = self.Prev_layer_H
        #self.cache["Pre_act"] = H
        return A , H

class Feedforward:

    def __init__(self,X_train, activation_fn, method, network):
        self.input = X_train
        self.activation_fn = activation_fn
        self.method = method
        self.network = network

    def Forward_prop(self):

        L = len(self.network)
        # print(L)

        for i in range(L):
        # print(i)
        # print("oth layer input" , self.input.shape)

            self.network[i].a, self.network[i].h = Pre_Feedforward(self.input, self.network[i].weight, self.network[i].bias, self.activation_fn[i]).Preactivation_cal()
            self.input = self.network[i].a
            # print("oth layer w" , a[i].weight.shape)
            # print("oth layer b" , a[i].bias.shape)
            # print("oth layer h" , a[i].h.shape)
            # print("oth layer a" , a[i].a.shape)


        return self.network
