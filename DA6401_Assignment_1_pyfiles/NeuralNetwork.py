
import tensorflow as tf
import numpy as np
from pickle import FALSE 
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
from tqdm import tqdm
from Loss_grad import *
from Activation_grad import *
from Feedforward import *
from Backprop import *
from Initilize import *
from Preprocessing import *
from optimizer import *
from Regularization import *
from NeuralNetwork import *
import math
import copy
import wandb

class NeuralNetwork:
    def __init__(self, loss_function, X_train, y_train, activation_fn, optimization_fn,
                 layers_dimensions, method, batch_size, epochs, val_ratio,
                 weight_decay=0, eta=0.01, beta=0.9, beta2=0.999,
                 regularization_fn="L2", grad_reglr_fn="L2_d"):

        self.loss_function = loss_function
        self.X_train = X_train
        self.y_train = y_train
        self.activation_fn = activation_fn
        self.method = method
        self.n = layers_dimensions
        self.optimization_fn = optimization_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_ratio = val_ratio
        self.weight_decay = weight_decay
        self.eta = eta
        self.epsilon = 1e-10
        self.beta = beta
        self.beta2 = beta2
        self.regularization_fn = regularization_fn
        self.grad_reglr_fn = grad_reglr_fn
        self.batches_number = max(1, self.X_train.shape[1] // self.batch_size)


        # Splitting data
        X_split, y_split, valX_split, valy_split = TrainValSplit(self.X_train, self.y_train, self.val_ratio).Apply_split()

        # Normalization & Encoding
        self.X_norm = Normalize(X_split).Norm_reshape()
        self.y_norm = OneHotEncoder(X_split, y_split).onehot_encode()
        self.valX_norm = Normalize(valX_split).Norm_reshape()
        self.valy_norm = OneHotEncoder(valX_split, valy_split).onehot_encode()

        # Ensure correct dimensions
        if self.X_norm.shape[1] != self.y_norm.shape[1]:
            raise ValueError(f"Mismatch in training data shapes: {self.X_norm.shape} vs {self.y_norm.shape}")

        if self.valy_norm is not None and self.valX_norm.shape[1] != self.valy_norm.shape[1]:
            raise ValueError(f"Mismatch in validation data shapes: {self.valX_norm.shape} vs {self.valy_norm.shape}")

    def trainNN(self):
        NNModel = GiveOptimizers(
            self.optimization_fn, self.loss_function, self.X_norm, self.y_norm,
            self.activation_fn, self.n, self.method, self.batch_size, self.epochs,
            self.valX_norm, self.valy_norm, self.weight_decay, self.eta,
            self.beta, self.beta2, self.regularization_fn, self.grad_reglr_fn
        ).apply_optimization()

        if NNModel is None:
            raise ValueError("Model initialization failed. `GiveOptimizers.apply_optimization()` returned None.")

        if not isinstance(NNModel, (list, tuple)) or len(NNModel) < 5:
            raise ValueError(f"Unexpected NNModel structure: {type(NNModel)} with length {len(NNModel)}")

        final_network = NNModel[0]
        overall_loss = NNModel[1]
        accuracy = NNModel[2]
        val_loss = NNModel[3]
        val_acc = NNModel[4]

        return final_network, overall_loss, accuracy, val_loss, val_acc