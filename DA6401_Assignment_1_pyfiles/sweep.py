# -*- coding: utf-8 -*-
"""Sweep.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GoNCulot2ID4Nmx2gPZ8th_2DHla3jtj
"""

import wandb
from tensorflow.keras.datasets import fashion_mnist

# Load data:
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# ✅ Sweep configuration matching all provided parameters:
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_hidden_layers': {'values': [3, 4, 5]},
        'hidden_layer_size': {'values': [64, 128]},
        'weight_decay': {'values': [0, 0.0005]},
        'learning_rate': {'values': [ 0.001]},  # Match large range from your example
        'optimizer': {'values': [ 'nag', 'rmsp', 'adam']},
        'batch_size': {'values': [100, 500, 1000]},  # Your example used 1000; also add variations
        'weight_init': {'values': ['Random', 'Xavier_U']},
        'activation_fn': {'values': ['sigmoid', 'tanh', 'relu', 'lrelu']}
    }
}

# ✅ Sweep training function using correct call signature:
def sweep_train():
    wandb.init()
    config = wandb.config

    # Create hidden layers and activation lists dynamically
    hidden_sizes = [config.hidden_layer_size] * config.num_hidden_layers
    activations = [config.activation_fn] * config.num_hidden_layers

    # Proper run name:
    wandb.run.name = f"hl_{config.num_hidden_layers}_bs_{config.batch_size}_ac_{config.activation_fn}"

    # Call trainer with your exact parameters and sweep values:
    trainer = NNTrainer(
        X_train=X_train,
        y_train=y_train,
        num_hidden_layers=config.num_hidden_layers,
        hidden_layer_sizes=hidden_sizes,
        activation_fns=activations,
        optimizer=config.optimizer,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        weight_init=config.weight_init,
        loss_fn="ce",
        reg_type="L2",
        weight_decay=config.weight_decay,
        val_ratio=0.1,
        method=config.weight_init,  # Use same value as weight_init for consistency
        beta=0.9,
        beta2=0.999,
        X_test=X_test,
        y_test=y_test
    )

    train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = trainer.train()

    wandb.log({
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'test_loss': test_loss,
        'test_accuracy': test_acc
    })

# ✅ Start the sweep:
sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_hyperparam_search")
wandb.agent(sweep_id, function=sweep_train, count =5)


wandb.finish()