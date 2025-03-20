import numpy as np
from Regularization import *

class CalLoss:
    def __init__(self, y, y_pred):
        self.y = y
        self.y_predicted = y_pred
        if y.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y shape is {self.y.shape}, y_predicted shape is {self.y_predicted.shape}")

class CrossEntropy(CalLoss):
    def give_celoss(self):
        epsilon = 1e-8  # Small value to prevent log(0)
        return -np.mean(self.y * np.log(self.y_predicted + epsilon))

    def Give_cegrad(self):
        epsilon = 1e-8  # Prevent division by zero
        grad = -self.y / (self.y_predicted + epsilon)
        return grad

class SquaredError(CalLoss):
    def give_seloss(self):
        return np.mean((self.y - self.y_predicted) ** 2)

    def Give_segrad(self):
        grad = -2 * (self.y - self.y_predicted)
        return np.clip(grad, -1, 1)  # Clip to prevent exploding gradients

class callloss(CalLoss):
    def __init__(self, loss_function, y, y_pred):
        self.loss_function = loss_function.lower()
        super().__init__(y, y_pred)

    def give_loss(self):
        if self.loss_function == 'ce':
            return CrossEntropy(self.y, self.y_predicted).give_celoss()
        elif self.loss_function == 'se':
            return SquaredError(self.y, self.y_predicted).give_seloss()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def give_gradloss(self):
        if self.loss_function == 'ce':
            return CrossEntropy(self.y, self.y_predicted).Give_cegrad()
        elif self.loss_function == 'se':
            return SquaredError(self.y, self.y_predicted).Give_segrad()
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")


class CalculateAllLoss:
    def __init__(self, X_train, y_predicted,network, y_train, primary_loss, weight_decay=0, regularisation_fn=None):
        self.y_predicted = y_predicted
        self.y_true = y_train
        self.network = network
        self.X_train = X_train
        self.loss_value = primary_loss
        self.weight_decay = weight_decay
        self.regularisation_fn= regularisation_fn
        self.calc_accuracy_loss()


    def overall_loss(self):
        """
        Calculates the total loss of the network.

        This includes:
        - The primary loss (e.g., Cross-Entropy Loss)
        - Optional regularization (like L2 regularization)

        Parameters:
        - network: The neural network model.
        - Y_pred: The predicted output from the network.
        - Y_true: The actual labels (ground truth).
        - loss_fn: The loss function to be used (e.g., CrossEntropy_loss).
        - weight_decay: A coefficient for regularization (default is 0, meaning no regularization).
        - regularisation_fn: A function for computing the regularization term (optional).

        Returns:
        - Total loss value.
        """
        # Get network predictions self.y_prediction from model
        total_loss = self.loss_value  # Compute primary loss

        if self.weight_decay > 0 and self.regularisation_fn:
            regularized_val = ApplyReg(self.regularisation_fn, self.network).do_reg()
            print(f"Reg value: {regularized_val}")
            total_loss += self.weight_decay * regularized_val # Add regularization term if applicable  # Compute total loss
        return total_loss




    def calc_accuracy_loss(self):
        """
        Computes the accuracy and loss for a given neural network.

        Parameters:
        - network: The neural network model.
        - X: Input data (features for prediction).
        - Y: Actual labels (ground truth values).
        - loss_fn: The loss function to be used.
        - weight_decay: Regularization strength (default is 0, meaning no regularization).
        - regularisation_fn: A function to compute the regularization term (optional).

        Returns:
        - accuracy: The percentage of correctly classified examples.
        - loss: The computed total loss.
        """

        total_loss = self.loss_value  # Compute primary loss

        if self.weight_decay > 0 and self.regularisation_fn:
            regularized_val = ApplyReg(self.regularisation_fn, self.network).do_reg()
            print(f"Reg value: {regularized_val}")
            total_loss += self.weight_decay * regularized_val

        # Ensure dimensions match between input and labels
        assert self.X_train.shape[1] == self.y_true.shape[1], "Mismatch in batch size between inputs and labels"

        # Compute accuracy by comparing predicted vs actual labels
        batch_size = self.X_train.shape[1]  # Number of examples
        correct_predictions = np.sum(np.argmax(self.y_predicted, axis=0) == np.argmax(self.y_true, axis=0))

        accuracy = correct_predictions / batch_size  # Compute accuracy as a fraction

        return accuracy , total_loss

