
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


class NNTrainer:
    def __init__(self, X_train, y_train, num_hidden_layers, hidden_layer_sizes, activation_fns,
                 optimizer, batch_size, learning_rate, epochs, weight_init, loss_fn,
                 reg_type="L2", weight_decay=0, val_ratio=0.2, method="default",
                 beta=0.9, beta2=0.999,
                 X_test=None, y_test=None):
        """
        Wrapper class to simplify training a NeuralNetwork model.
        Now it dynamically constructs the layer dimensions and activation functions.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_fns = activation_fns
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_init = weight_init
        self.loss_fn = loss_fn
        self.reg_type = reg_type
        self.weight_decay = weight_decay
        self.val_ratio = val_ratio
        self.method = method
        self.beta = beta
        self.beta2 = beta2
        self.X_test = X_test
        self.y_test = y_test

        # Automatically construct layer dimensions
        self.layers_dimensions = self._construct_layers()
        self.activation_fn = self._construct_activations()

        # Data Preprocessing
        self._prepare_data()

    def _construct_layers(self):
        """Constructs layer dimensions based on the number of hidden layers and their sizes."""
        input_size = 784  # Fashion-MNIST input size
        output_size = 10   # Number of classes in Fashion-MNIST
        return [input_size] + self.hidden_layer_sizes[:self.num_hidden_layers] + [output_size]

    def _construct_activations(self):
        """Constructs activation functions for each hidden layer, ending with softmax for classification."""
        return self.activation_fns[:self.num_hidden_layers] + ["softmax"]


    def _prepare_data(self):
        """Splits and normalizes train/validation/test data."""
        self.X_train, self.y_train, self.X_val, self.y_val = TrainValSplit(
            self.X_train, self.y_train, self.val_ratio).Apply_split()

        self.X_norm = Normalize(self.X_train).Norm_reshape()
        self.y_norm = OneHotEncoder(self.X_train, self.y_train).onehot_encode()
        self.X_val_norm = Normalize(self.X_val).Norm_reshape()
        self.y_val_norm = OneHotEncoder(self.X_val, self.y_val).onehot_encode()

        if self.X_test is not None and self.y_test is not None:
            self.X_test_norm = Normalize(self.X_test).Norm_reshape()
            self.y_test_norm = OneHotEncoder(self.X_test, self.y_test).onehot_encode()
        else:
            self.X_test_norm = None
            self.y_test_norm = None

    def train(self):
        """Trains the neural network and returns train/val/test metrics."""
        model = NeuralNetwork(
            loss_function=self.loss_fn,
            X_train=self.X_train, y_train=self.y_train,
            activation_fn=self.activation_fn,
            optimization_fn=self.optimizer,
            layers_dimensions=self.layers_dimensions,
            method=self.method, batch_size=self.batch_size,
            epochs=self.epochs, val_ratio=self.val_ratio,
            weight_decay=self.weight_decay, eta=self.learning_rate,
            beta=self.beta, beta2=self.beta2,
            regularization_fn=self.reg_type,
            grad_reglr_fn=f"{self.reg_type}_d",
        )

        # Train the model
        final_network, overall_loss, train_accuracy, val_loss, val_accuracy = model.trainNN()

        # If test data is provided, evaluate on test set
        test_loss, test_accuracy = None, None
        if self.X_test_norm is not None and self.y_test_norm is not None:
            y_test_predicted = Feedforward(self.X_test_norm, self.activation_fn, self.method, final_network).Forward_prop()
            y_test_pred = np.array(y_test_predicted[-1].h)
            loss = callloss(self.loss_fn, self.y_test_norm, y_test_pred).give_loss()
            test_loss_obj = CalculateAllLoss(X_train=self.X_test_norm, y_predicted=y_test_pred, network=final_network,
                                             y_train=self.y_test_norm, primary_loss=loss, weight_decay=self.weight_decay,
                                             regularisation_fn=self.reg_type)
            test_accuracy, test_loss = test_loss_obj.calc_accuracy_loss()

        return overall_loss, train_accuracy, val_loss, val_accuracy, test_loss, test_accuracy


trainer = NNTrainer(
    X_train=X_train,
    y_train=y_train,
    num_hidden_layers=5,  # Specify the number of hidden layers
    hidden_layer_sizes=[128, 64, 32],  # Specify possible sizes per hidden layer
    activation_fns=["tanh", "sigmoid", "tanh"],  # Specify possible activations
    optimizer="adam",
    batch_size=1000,
    learning_rate=0.01,
    epochs=5,
    weight_init="Xavier_U",
    loss_fn="ce",
    reg_type="L2",
    weight_decay=0.0005,
    val_ratio=0.1,
    method="Xavier_U",
    beta=0.9,
    beta2=0.999,
    X_test=X_test,
    y_test=y_test
)

train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = trainer.train()