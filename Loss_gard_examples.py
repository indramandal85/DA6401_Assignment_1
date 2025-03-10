{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "5f6897d3-0f77-488e-921c-009c7e463692",
      "metadata": {
        "id": "5f6897d3-0f77-488e-921c-009c7e463692"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "# Loss Functions\n",
        "class Loss:\n",
        "    \"\"\"Handles different loss functions with easy selection by name.\"\"\"\n",
        "\n",
        "    def __init__(self, loss_name=\"cross_entropy\"):\n",
        "        self.loss_name = loss_name.lower()\n",
        "\n",
        "    def compute_loss(self, y, y_hat):\n",
        "        if self.loss_name == \"cross_entropy\":\n",
        "            return self.cross_entropy(y, y_hat)\n",
        "        elif self.loss_name == \"mse\":\n",
        "            return self.mse(y, y_hat)\n",
        "        else:\n",
        "            raise ValueError(\"Invalid loss name. Choose 'cross_entropy' or 'mse'.\")\n",
        "\n",
        "    @staticmethod\n",
        "    def cross_entropy(y, y_hat):\n",
        "        \"\"\"Computes cross-entropy loss.\"\"\"\n",
        "        epsilon = 1e-12\n",
        "        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)\n",
        "        return -np.mean(np.sum(y * np.log(y_hat), axis=1))\n",
        "\n",
        "    @staticmethod\n",
        "    def mse(y, y_hat):\n",
        "        \"\"\"Computes mean squared error loss.\"\"\"\n",
        "        return np.mean((y - y_hat) ** 2)\n",
        "\n",
        "# Gradient of Loss Functions\n",
        "class LossGradient:\n",
        "    \"\"\"Handles different loss function gradients with easy selection by name.\"\"\"\n",
        "\n",
        "    def __init__(self, loss_name=\"cross_entropy\"):\n",
        "        self.loss_name = loss_name.lower()\n",
        "\n",
        "    def compute_gradient(self, y, y_hat):\n",
        "        if self.loss_name == \"cross_entropy\":\n",
        "            return self.cross_entropy_grad(y, y_hat)\n",
        "        elif self.loss_name == \"mse\":\n",
        "            return self.mse_grad(y, y_hat)\n",
        "        else:\n",
        "            raise ValueError(\"Invalid loss name. Choose 'cross_entropy' or 'mse'.\")\n",
        "\n",
        "    @staticmethod\n",
        "    def cross_entropy_grad(y, y_hat):\n",
        "        \"\"\"Computes gradient of cross-entropy loss.\"\"\"\n",
        "        epsilon = 1e-12\n",
        "        y_hat = np.clip(y_hat, epsilon, 1 - epsilon)\n",
        "        return -y / y_hat\n",
        "\n",
        "    @staticmethod\n",
        "    def mse_grad(y, y_hat):\n",
        "        \"\"\"Computes gradient of mean squared error loss.\"\"\"\n",
        "        return -2 * (y - y_hat) / y.shape[0]\n",
        "\n",
        "# Wrapper Class for Easy Use\n",
        "class LossHandler:\n",
        "    \"\"\"Unified interface to compute loss and gradient using different loss functions.\"\"\"\n",
        "\n",
        "    def __init__(self, loss_name=\"cross_entropy\"):\n",
        "        self.loss_function = Loss(loss_name)\n",
        "        self.loss_gradient = LossGradient(loss_name)\n",
        "\n",
        "    def get_loss(self, y, y_hat):\n",
        "        return self.loss_function.compute_loss(y, y_hat)\n",
        "\n",
        "    def get_gradient(self, y, y_hat):\n",
        "        return self.loss_gradient.compute_gradient(y, y_hat)\n",
        "\n",
        "# Example Usage:\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "f8eec849-fb45-4f11-8790-01de712bf7a4",
      "metadata": {
        "id": "f8eec849-fb45-4f11-8790-01de712bf7a4",
        "outputId": "1ee63200-ac39-426c-896e-5a426bdd945a"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "array([[0, 1, 0],\n",
              "       [1, 0, 0]])"
            ]
          },
          "execution_count": 3,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "y_true = np.array([[0, 1, 0,], [1, 0, 0]])  # One-hot encoded labels\n",
        "y_true\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "f93d41ae-e7f6-46b3-976c-6c889c6e2670",
      "metadata": {
        "id": "f93d41ae-e7f6-46b3-976c-6c889c6e2670"
      },
      "outputs": [],
      "source": [
        "y_true = np.array([[0, 1, 0,], [1, 0, 0]])\n",
        "y_pred_ce = np.array([[0.2, 0.7, 0.1], [0.8, 0.1, 0.1]])  # Predicted probabilities\n",
        "\n",
        "y_true_mse = np.array([3.0, 5.0, 2.5])\n",
        "y_pred_mse = np.array([2.8, 5.2, 2.7])\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "3bb211c9-b2d6-4462-b56d-e132e02e55e0",
      "metadata": {
        "id": "3bb211c9-b2d6-4462-b56d-e132e02e55e0",
        "outputId": "90b8289a-4fba-4c7e-f25d-0c1be9b4fc03"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "0.2899092476264711\n",
            "Cross Entropy Gradient: [[ 0.         -1.42857143  0.        ]\n",
            " [-1.25        0.          0.        ]]\n"
          ]
        }
      ],
      "source": [
        "# Cross Entropy Loss (Default)\n",
        "ce_loss_handler = LossHandler()\n",
        "print(ce_loss_handler.get_loss(y_true, y_pred_ce))\n",
        "print(\"Cross Entropy Gradient:\", ce_loss_handler.get_gradient(y_true, y_pred_ce))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "3056545e-c08b-4d58-86f5-c3e66ffdb7b9",
      "metadata": {
        "id": "3056545e-c08b-4d58-86f5-c3e66ffdb7b9",
        "outputId": "dca67d6a-1296-44e4-af07-4fbb8cff5b8d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "MSE Loss: 0.04000000000000007\n",
            "MSE Gradient: [-0.13333333  0.13333333  0.13333333]\n"
          ]
        }
      ],
      "source": [
        "\n",
        "# MSE Loss\n",
        "mse_loss_handler = LossHandler(\"mse\")\n",
        "print(\"MSE Loss:\", mse_loss_handler.get_loss(y_true_mse, y_pred_mse))\n",
        "print(\"MSE Gradient:\", mse_loss_handler.get_gradient(y_true_mse, y_pred_mse))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "cc5143ba-2218-45a2-99ef-4af2a975980f",
      "metadata": {
        "id": "cc5143ba-2218-45a2-99ef-4af2a975980f"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python [conda env:base] *",
      "language": "python",
      "name": "conda-base-py"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.12.4"
    },
    "colab": {
      "provenance": []
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}