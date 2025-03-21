
#  Problem Statement:
## 📚 Question 1
## Download the Fashion-MNIST dataset and plot one sample image for each class in a clean grid layout.


To deploy this project run

```bash
   from keras.datasets import fashion_mnist
```


## 🛠️ Methodology:
Data Loading:

The FashionMNISTLoader class is created to load training data and store predefined class labels.
Sample Selection:

For each of the 10 classes, the first occurrence of that class is identified and extracted as a sample image.
Visualization:

A 2x5 grid plot is created using matplotlib to display one representative image per class.
Each subplot is labeled with its corresponding class name for easy identification.
## Variables & Components:



| Variable / Class   |	Purpose                |
| :------- | :------------------------- |
| `FashionMNISTLoader` | Loads the Fashion-MNIST dataset and prepares sample images.|
| `self.class_labels` |List containing names of the 10 Fashion-MNIST classes. |
| `get_sample_images()` | Function to extract one sample image per class. |
| `ImagePlotter` | Static class to handle plotting and visualization.|
| `plot_images(image_data)` | 	Method to arrange and plot images in a grid with class labels. |
| `FashionMNISTVisualizer` | Integrates loading and plotting for easy visualization. |
| `visualizer.visualize_samples()` | Calls the functions to generate and show the final output.|



## ✅ Output:
The final output is a 2x5 grid of grayscale images, each labeled with its corresponding class name from the Fashion-MNIST dataset, showcasing clear class-wise representation.

 #### Solution is in the ED24S014_Assignment_1.ipynb file.
 * The link of the Github Repository & the wandb :
 ### 🔗 Links
https://github.com/indramandal85/DA6401_Assignment_1.git

## 📚 Question 2
## Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.
This solution presents a full custom pipeline for building and forward-predicting a neural network from scratch using NumPy, demonstrating key machine learning preprocessing steps, network initialization, feedforward propagation, and prediction on the Fashion MNIST dataset. Below is the structured explanation:
## 🛠️ Methodology:
### Data Preprocessing:

* Convert the categorical labels into one-hot encoded vectors.
* Normalize image data for stable and efficient training.
* Optionally split training data into training and validation sets for evaluation.
### Network Construction:

* Initialize weights and biases layer by layer using different initialization methods (e.g., Xavier Uniform).
* Define each layer’s forward pass with pre-activation and post-activation computations.
### Activation Functions:

* Define multiple activation functions (sigmoid, relu, leaky relu, tanh, softmax) along with their derivatives for flexibility across hidden and output layers.
### Feedforward Prediction:

* Pass normalized input through each initialized layer using matrix multiplication and activation.
* The final layer outputs prediction probabilities (in this case, for 10 classes in Fashion MNIST).

## ✅ Process Steps:
### 1. Data Preprocessing

#### Normalization: 
* The `Normalize` class reshapes the input image data from `(N, 28, 28)` to `(784, N)` and scales pixel values between 0 and 1 by dividing by 255.

#### One-Hot Encoding: 
* The `OneHotEncoder` class transforms label vectors into one-hot encoded matrices with shape `(10, N)`  for 10 classes.

### 2. Weight and Bias Initialization
* The `Weight_bias` and `InitializeWeights` classes handle the creation of a neural network structure:
* The weight initialization methods supported:
   * Xavier Normal (`Xavier_N`),
   * Xavier Uniform (`Xavier_U`),
   * He Normal (`He_N`),
   * He Uniform (`He_U`),
   * Random small initialization.
* In this solution, Xavier Uniform initialization is used: 
```bash
   self.weight = np.random.uniform((-a), a,( self.o_size,self.i_size))
```
* Biases are initialized to zeros for all layers.

### 3.  Activation Functions
* Multiple activation functions are implemented through classes:
   * Sigmoid, ReLU, Leaky ReLU, Tanh, and Softmax.
   * Each activation function has its derivative defined for future training use.
* The `apply_activation` class chooses the appropriate function dynamically based on layer configuration.
### 4. Feedforward Pass
* The feedforward mechanism is done through:
   * The Pre_Feedforward class that calculates:
      * Pre-activation value:
      
         `A = W * input + b`
      * Normalizes A and then applies the selected activation function.
   * The `Feedforward `class loops through each layer and updates the input for the next layer using these calculations.
   * The final output layer produces class probabilities through softmax.

## ✅ Code Implementation
* Loading dataset:
```bash
   (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
```
* Normalization & One-Hot Encoding:
```bash
   X_new = Normalize(X_train).Norm_reshape()
y_new = OneHotEncoder(X_train, y_train).onehot_encode()
```
* Network Initialization (with Xavier Uniform):
```bash
   Initial_network = Weight_bias(layer_dimension=[784, 128, 64, 10],
                              activation_function=['sigmoid', 'relu', 'softmax'],
                              y_train=y_new,
                              method="Xavier_U").Init_network()
```
* Feedforward Prediction:
```bash
   Predict = Feedforward(X_train=X_new,
                      activation_fn=['sigmoid', 'relu', 'softmax'],
                      method="Xavier_U",
                      network=Initial_network).Forward_prop()
```
* Output (Prediction):
```bash
   print("Prediction of output of 10 classes in probability distribution form with Xavier Uniform initialized weights and biases: \n\n", Predict[2].h)
```
 #### Solution is in the ED24S014_Assignment_1.ipynb file.
 * The link of the Github Repository & the wandb :
 ### 🔗 Links
https://github.com/indramandal85/DA6401_Assignment_1.git

https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg

## 📚 Question 3
## Neural Network Backpropagation with Multiple Optimizers
Implement the backpropagation algorithm for a neural network with support for the following optimization techniques:

### ✅ Introduction:
This part focuses on the implementation of neural network backpropagation with support for multiple optimization algorithms. The objective is to train a feedforward neural network on given training data using advanced optimization techniques beyond standard gradient descent, allowing for improved convergence, stability, and generalization.

The implemented optimizers include:

* Stochastic Gradient Descent (SGD)
* Momentum-based Gradient Descent
* Nesterov Accelerated Gradient (NAG)
* RMSProp
* ADAM
Each optimizer is coded from scratch to demonstrate the underlying mathematical concepts and algorithms used in modern deep learning frameworks.

### 🛠️ Methodology:

#### The methodology involves:

* Defining a feedforward neural network architecture with customizable layer dimensions and activation functions.
* Implementing backpropagation for gradient calculation based on the chosen loss function.
* Applying optimization algorithms to update weights and biases.
* Supporting batch processing and dynamic learning rate adjustments.
* Recording loss history, training accuracy, validation accuracy, and learning rate trends for analysis.
#### Mathematical foundations include:

* Weight updates using gradients (for SGD)
* Velocity-based updates (Momentum, NAG)
* Adaptive learning rates using moving averages (RMSProp)
* Combined momentum and RMSProp update rules (ADAM)

### ✅ Way of Implementation: 
* Base Class: `Optimizer`

    Initializes shared parameters including learning rate, weight decay, batch size, and regularization settings.

* `SGD` Class

    * Method: Gradient_descent()
    * Performs standard stochastic gradient descent.

* `MGD` (Momentum Gradient Descent) Class

    * Method: momentum_GD()
    * Uses velocity terms for accelerated convergence.

* `NAG` (Nesterov Accelerated Gradient) Class

    * Method: Nesterov_AGD()
    * Updates parameters with a look-ahead mechanism.
* `RMSProp` Class

    * Method: rms_GD()
    * Maintains moving average of squared gradients for adaptive updates.
* `ADAM` Class

    * Method: adam_GD()
    * Combines momentum and RMSProp with bias correction and dynamic learning rate scheduling.

* `GiveOptimizers`

Dynamically selects and applies the desired optimization technique by invoking the corresponding method.

### ✅ Implementation Example (Code Reference)
```bash
   from optimizer import GiveOptimizers

optimizer = GiveOptimizers(
    optimization_fn='adam',
    loss_function='cross_entropy',
    X_train=X_train,
    y_train=y_train,
    activation_fn='relu',
    layers_dimensions=[784, 128, 64, 10],
    method='he',
    batch_size=64,
    epochs=50,
    validX_train=X_val,
    validy_train=y_val,
    eta=0.001,
    weight_decay=0.01
)

loss_history, training_history, eta_history, validation_history, trained_model = optimizer.apply_optimization()

```

### ✅ Training Logs Example (ADAM Optimizer)

```bash
Epoch 1/50 | Train Acc: 0.78 | Val Acc: 0.75 | Train Loss: 0.5634
Epoch 10/50 | Train Acc: 0.84 | Val Acc: 0.82 | Train Loss: 0.3124
Epoch 25/50 | Train Acc: 0.84 | Val Acc: 0.81 | Train Loss: 0.2102
Epoch 50/50 | Train Acc: 0.81 | Val Acc: 0.80 | Train Loss: 0.1287


```

 #### Full coding solution is in the ED24S014_Assignment_1.ipynb file.
 * The link of the Github Repository & the wandb :
 ### 🔗 Links
https://github.com/indramandal85/DA6401_Assignment_1.git

https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw

## 📚 Question 4
## Hyperparameter Tuning for Fashion MNIST Using W&B Sweeps
The objective was to efficiently identify the best set of hyperparameters (like number of layers, learning rate, activation functions, optimizer, etc.) for maximum validation accuracy and minimum validation loss using sweep configuration in wandb.

### 🛠️ Methodology:
#### Dataset Preparation
* Dataset: Fashion MNIST (60,000 training samples, 10,000 test samples)
* Preprocessing:
    * Normalization of pixel values between 0 and 1
    * Splitting the training set into 90% training and 10% validation
#### Model Architecture (Reference from Code)
The model was designed dynamically based on hyperparameters from each sweep iteration.

* Hyperparameter-defined hidden layers (3–5 layers)
* Layer size (32, 64, or 128 units)
* Activation functions: sigmoid, tanh, or ReLU
* L2 regularization (weight decay) applied according to sweep config
* Output layer with softmax activation for multi-class classification

### ✅ Implementation Strategy
#### W&B Sweep Setup
* Defined sweep configurations with `wandb.sweep()` using the following parameter space:
    * Epochs: 5 or 10
    * Learning rate: 1e-3, 1e-4
    * Optimizers: SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
    * Number of hidden layers: 3, 4, 5
    * Layer sizes: 32, 64, 128
    * Batch sizes: 16, 32, 64
    * Weight initialization: random, Xavier
    * ctivation function: sigmoid, tanh, ReLU
    * Weight decay: 0, 0.0005, 0.5
#### Training Process (Code Reference):

* The notebook includes a function `NNTrainer()` that:
    * Builds the model using dynamic hyperparameters
    * Compiles it with selected optimizer, learning rate, and loss function
    * Trains over the specified epochs and batch size
    * Logs training and validation loss/accuracy metrics to W&B
#### Running Sweeps
* Used the `wandb.agent()` function to launch multiple runs from the defined sweep
* Each run used a random combination of hyperparameters from the sweep config
### ✅ Code Structure:

| Section   |	Description              |
| :------- | :------------------------- |
| Data Loading & Preprocessing | Load Fashion MNIST dataset, normalize, split into train/validation/test|
|  Model Building Function|Dynamically builds model layers, hidden units, and activations using wandb.config parameters |
| Sweep Configuration | Dictionary with search spaces for each hyperparameter |
| Training Function| Compiles the model with optimizer, learning rate, and loss function, and fits using the `NeuralNetwork` |
| W&B Integration| 	wandb.init for experiment tracking and wandb.agent for running multiple sweep configurations |

### ✅ Sweep Configuration:
The sweep was performed with the following configurations:
```bash
   sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_hidden_layers': {'values': [3, 4, 5]},
        'hidden_layer_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [0.001, 0.0001]},
        'optimizer': {'values': ["sgd", 'momentum', 'nag', 'rmsp', 'adam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['Random', 'Xavier_U']},
        'activation_fn': {'values': ['sigmoid', 'tanh', 'relu']}
    }
}

```
### ✅ Training Function:
The sweep training function was written as follows, where each experiment logs key metrics to wandb:
```bash
def sweep_train():
    wandb.init()
    config = wandb.config

    hidden_sizes = [config.hidden_layer_size] * config.num_hidden_layers
    activations = [config.activation_fn] * config.num_hidden_layers

    wandb.run.name = f"hl_{config.num_hidden_layers}_bs_{config.batch_size}_ac_{config.activation_fn}"

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
        method=config.weight_init,
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
```
The sweep was executed as:
```bash
sweep_id = wandb.sweep(sweep_config, project="DA6401_A1_Q4")
wandb.agent(sweep_id, function=sweep_train)
```

### 📊 Results and Outputs:
#### 🔎 Observations
* The best-performing runs consistently featured:
    * 4 hidden layers
    * Layer size of 128
    * Learning rate = 1e-3
    * Optimizer: Adam 
    * Weight initialization: Xavier
    * Activation: ReLU
    * Lower batch sizes (16 or 32) performed better in terms of convergence.
    * Higher L2 regularization (0.5) tended to over-regularize the model and increased loss.

 #### Full coding solution is in the DA5401_Assignment1_ED24S014.ipynb file in github.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link :   https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw
## 📚 Question 5 
### Best Accuracy Plot:
* After running the sweep, I used the wandb.ai "Add Panel to Report" feature to generate the test accuracy plot.
* The plot shows the accuracy of various models across multiple configurations.

 #### Full Generated Accuracy Plot is in the wandb report.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link :  https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw
## 📚 Question 6
### Analysis and Inferences : 
Using the parallel coordinates plot and correlation analysis from wandb.ai, I derived the following insights:

#### 🔎 Key Observations:
* Learning Rate has the most significant impact on accuracy. Very small learning rates (0.0001) tended to underperform, while moderate learning rates (0.001) consistently achieved higher accuracy.
* Optimizers:
    * Adam optimizer outperformed SGD and other variants in most cases.
    * SGD and Momentum configurations resulted in lower accuracies, often under 65%.
* Weight Decay: Extremely high weight decay values negatively impacted accuracy. Low or zero weight decay performed better.
* Hidden Layer Size: Larger hidden layers (64 or 128) combined with the right learning rate and optimizer gave better performance.
* Activation Functions: The relu activation function dominated other choices (sigmoid and tanh).
* Batch Size: A moderate batch size (32 or 64) worked better in combination with Adam and appropriate learning rates.

#### 🎯 Recommended Configuration to Approach 95% Accuracy:
| Parameter  |	Recommended Value          |
| :------- | :------------------------- |
| Optimizer | `ADAM` |
| Learning Rate |`0.001` |
| Hidden Layer Size| `128` |
| Weight Decay | `0.0005` |
| Activation Function | `RELU`  |
| Number of Hidden Layers | `4` or `5` |
| Batch Size | `32` |

### ✅ Visualizations :
 #### Full Generated Parallel Coordinates Plot and Parameter Importance Visualization are in the wandb report.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link :  https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw
## 📚 Assignment Question 7: 
### Confusion Matrix Generation: 
Here in this part we generates a confusion matrix to evaluate the model's classification accuracy. The implementation includes model training, evaluation, and visualization using Matplotlib, Seaborn, and Weights & Biases (WandB) for logging and tracking results.

### 🛠️ Methodology:
#### Dataset & Preprocessing

* The dataset used is Fashion-MNIST, which consists of 10 classes of fashion items.
* Data normalization is performed to scale pixel values, ensuring better convergence.
* One-hot encoding is applied to labels for multi-class classification.
#### Neural Network Architecture

* The model is a feedforward neural network with the following parameters:
    * 5 hidden layers
    * Hidden layer sizes: [128, 64, 32]
    * Activation functions: ["tanh", "sigmoid", "tanh"]
    * Optimizer: Adam
    * Loss function: Cross-Entropy
    * Regularization: L2 with weight decay of 0.0005
    * Weight Initialization: Xavier
    * Epochs: 5
    * Batch Size: 32
#### Training & Evaluation

* The neural network is trained using the above configuration.
* A confusion matrix is generated to assess classification accuracy by comparing true labels vs. predicted labels on the test dataset.
* The model predictions are stored, and the confusion matrix is normalized to visualize 
misclassifications effectively.
#### Logging & Visualization with WandB

* The confusion matrix is plotted using Seaborn with color-coded misclassifications.
* Green diagonal patches highlight correct classifications, while red shades represent misclassifications.
* The confusion matrix is logged to Weights & Biases (WandB) for detailed tracking and visualization.

### ✅ Code Implementation :
#### Confusion Matrix Plotting Class
The `plot_confusion_matrix` class is used to compute and visualize the confusion matrix after training the neural network. The following code initializes and executes the confusion matrix plotting function: 
```bash
   # Initialize WandB for logging
run = wandb.init(project="DA6401_Q7", reinit=True)

# Create and plot the confusion matrix
plot = plot_confusion_matrix(
    X_train=X_train,
    y_train=y_train,
    num_hidden_layers=5,  
    hidden_layer_sizes=[128, 64, 32],  
    activation_fns=["tanh", "sigmoid", "tanh"],  
    optimizer="adam",
    batch_size=10000,
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
plot.confusion_matrix()

# Finish WandB run
run.finish()

```
 #### Full Generated Confusion matrix Plot are  in the DA5401_Assignment1_ED24S014.ipynb file in github also in the wandb report.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link : https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw## 📚 Question 8
### Comparison of Cross Entropy Loss and Squared Error Loss :
Here in this part we explored and compared the performance of two widely used loss functions — Cross Entropy Loss and Squared Error Loss — in training deep learning models. While Cross Entropy Loss is generally preferred for classification problems, Squared Error Loss is more common for regression tasks. The objective was to observe how both loss functions influence model performance and validate findings with results and plots logged to Weights & Biases (wandb.ai).

### 🛠️ Methodology:
#### Loss Functions Setup:
* Implemented two variants of the loss function in the neural network training pipeline:
    * Cross Entropy (denoted as `ce`)
    * Squared Error (denoted as `Se`) 

####Hyperparameter Sweeps:

Here in this part i used the same sweep configuration as of question 4. But here i used both the loss functions in the sweep `ce` and `Se` .

#### Plot Generation to Compare Loss Functions
```bash
import wandb
api = wandb.Api()
runs = api.runs("DL_Assignment_8")

data = []
for run in runs:
    config = run.config
    val_acc = run.summary.get("val_accuracy")
    loss_fn = config.get("loss_fn")
    data.append((loss_fn, val_acc))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(data, columns=["Loss Function", "Validation Accuracy"])
sns.boxplot(x="Loss Function", y="Validation Accuracy", data=df)
plt.title("Cross Entropy vs Squared Error on Validation Accuracy")
plt.show()
```

### 🔎 Results & Observations: 
* The generated plots (both from local submission files and on wandb.ai) show that:

    * Cross Entropy Loss consistently results in higher validation accuracy compared to Squared Error Loss for classification tasks.
    * Squared Error Loss often struggled to converge or achieved lower accuracy, reinforcing the theoretical understanding that it is not ideal for classification problems.
* The detailed confusion matrix and training/validation performance plots are also logged on the wandb platform.

 #### Full Generated Plots are  in the DA5401_Assignment1_ED24S014.ipynb file in github also in the wandb report.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link : https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw
## 📚 Question 9
### Link of the Github Repo is given after every question's solution.

## 📚 Question 10
Based on the learnings of working using the Fashion Mnist dataset, here are 3 optimized configurations i could prefer: 

### Configuration 1 :
Rationale: Replicate the best Fashion-MNIST setup but simplify for MNIST.

Hidden Layers: 3 layers (sizes: [128, 64, 32])

Activation: ReLU (faster convergence for simpler data)

Optimizer: Adam (adaptive learning rate)

Learning Rate: 0.001 (lower than Fashion-MNIST for stability)

Batch Size: 64

Weight Decay: 0.0001 (mild L2 regularization)

Epochs: 10

### Configuration 2 :
Rationale: Reduce overkill complexity from Fashion-MNIST experiments.

Hidden Layers: 2 layers (sizes: [64, 32])

Activation: Sigmoid (smoother gradients for digit classification)

Optimizer: SGD + Momentum (momentum 0.9, learning rate 0.01)

Batch Size: 128

Regularization: Dropout 0.2 (prevent overfitting)

Epochs: 15

### Configuration 3 :
Rationale: Test if Fashion-MNIST regularization strategies generalize.

Hidden Layers: 4 layers (sizes: [256, 128, 64, 32])

Activation: tanh (matches Fashion-MNIST’s best setup)

Optimizer: AdamW (Adam + decoupled weight decay)

Learning Rate: 0.001

Weight Decay: 0.005 (stronger than Fashion-MNIST)

Batch Size: 256

Epochs: 20


#### Full code examples and plots are  in the DA5401_Assignment1_ED24S014.ipynb file in github also in the wandb report.
 #### The link of the Github Repository & the wandb report link :
 ### 🔗 Links
* Github Link : https://github.com/indramandal85/DA6401_Assignment_1.git
* wandb Link : https://wandb.ai/ed24s014-indian-institute-of-technology-madras/DL_A1/reports/DA6401-Assignment-1--VmlldzoxMTkwODYyMg?accessToken=e06ggims6l2xtj9i1y4kms8ja8bhn4f8zkvcrdg3wqol0tgkqqyycnjqld2o71mw