import numpy as np
from Activation_grad import *
from Loss_grad import *
from tqdm import tqdm
import math

class Backpropagation:

    def __init__(self, loss_function, X_train, y_train, y_pred, network, weight_decay, batch, batch_size, activation_fn):
        self.loss_function = loss_function
        self.X_train = X_train
        self.y_train = y_train
        self.y_predicted = y_pred
        self.network = network
        self.weight_decay = weight_decay
        self.batch = batch
        self.batch_size = batch_size
        self.activation_fn = activation_fn


    def backward_propagation(self):
      L = len(self.network)

      assert(self.y_train.shape[1] == self.y_predicted.shape[1])
      # print("y_pred_batch shape : " , self.y_pred_batch.shape)
      # print("y_true_batch shape : " , self.y_true_batch.shape)


      self.network[-1].d_a = callloss(self.loss_function,self.y_train, self.y_predicted).give_gradloss()
      # print("network -[-1].d_a : " , self.network[-1].d_a.shape)
      A_k = apply_activation(self.activation_fn[-1], self.network[-1].h).do_activation_derivative()
      # print("shape A_ k : " , A_k.shape)
      self.network[-1].d_h = self.network[-1].d_a * A_k
      # print("network -[-1].d_h : " , self.network[-1].d_a.shape)

      self.network[-1].d_w = self.network[-1].d_h @ self.network[-2].a.T  + self.weight_decay * self.network[-1].weight
      # print("network -[-1].d_w : " , self.network[-1].d_w.shape)
      d_b = -np.sum(self.network[-1].d_h, axis = 1)
      self.network[-1].d_b = d_b.reshape(-1 , 1)
      # print("network -[-1].d_b : " , self.network[-1].d_b.shape)



      for k in range(L-2,0,-1):
          # print(f"No of layers rotation {k}")

          self.network[k].d_h = self.network[k + 1].weight.T @ self.network[k + 1].d_a
          # print(f"shape self.network-{k}.d_h : " , self.network[k].d_h.shape)
          act_derv =  apply_activation(self.activation_fn[k], self.network[k].a)
          self.network[k].d_a = self.network[k].d_h * act_derv.do_activation_derivative()
          # print(f"shape self.network-{k}.d_a : " , self.network[k].d_a.shape)

          self.network[k].d_w = self.network[k].d_a @ self.network[k-1].h.T  + self.weight_decay * self.network[k].weight
          # print(f"shape self.network-{k}.d_w : " , self.network[k].d_w.shape)
          derv_bias = -np.sum(self.network[k].d_a, axis=1)
          self.network[k].d_b = derv_bias.reshape(-1 , 1)
          # print(f"shape self.network-{k}.d_b : " , self.network[k].d_b.shape)

      # print(f"shape self.network-{0}.d_a : " , self.network[0].d_a.shape)
      d_a = self.network[0].d_a[:, self.batch*self.batch_size : (self.batch+1)*self.batch_size]
      # print(f"shape self.network-{0}.d_a : " , d_a.shape)
      self.network[0].d_w = np.dot(d_a , self.X_train.T) + self.weight_decay * self.network[0].weight
      # print("network -[0].d_w : " , self.network[0].d_w.shape)
      self.network[0].d_b = np.sum(self.network[0].d_a, axis=1, keepdims = True)
      # print("network -[0].d_b : " , self.network[0].d_b.shape)


      return self.network




