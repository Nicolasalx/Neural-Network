# Usage
```
./my_torch_generator NEURAL_NETWORK.conf NUMER_OF_GENERATED_NN ...
```
```
./my_torch_analyzer --predict [--color] NEURAL_NETWORK.nn EVALUATION_FILE
./my_torch_analyzer --train [--save SAVEFILE] NEURAL_NETWORK.nn TRAINING_FILE
./my_torch_analyzer --train_stat [NEURAL_NETWORK.nn ...] TRAINING_FILE EVALUATION_FILE
```

## How to create a neural network

First, you will have to create a .conf that will contain the parameters of the neural network.  
Here is the structure of a .conf file:

```
learning_rate, min_learning_rate, max_learning_rate, momentum
neurons_per_layer
```

Here is an example of a .conf file with a neural network with those parameters:
- A **learning rate** of 0.1
- A **minimum learning rate** of 0.0001
- A **maximum learning rate** of 1
- A **momentum** of 0.9
- A neural network with **4 layers** with **16**, **12**, **8**, and **4 neurons**.

```
> cat my_neural_network.conf
0.1, 0.0001, 1, 0.9
16, 12, 8, 4
```

Now that we have created the configuration for our neural network, we can generate it !

```
> ./my_torch_generator my_neural_network.conf 1
```

This command will generate a my_neural_network.nn file that will contain the neural network.
Here is what the file will look like:

```
learning_rate: 0.1
min_learning_rate: 0.0001
max_learning_rate: 1
momentum: 0.9
nb_layer: 4
nb_neuron: 16
nb_neuron: 12
    nb_weight: 16, -0.378762, 0.425692, 0.172761, 0.329498, ...
    nb_weight_momentum: 16, 0, 0, 0, ...
    bias: 0.294357
    bias_momentum: 0
    ...

nb_neuron: 8
    nb_weight: 12, 0.453507, 0.400997, 0.976021, ...
    nb_weight_momentum: 12, 0, 0, 0, ...
    bias: -0.451216
    bias_momentum: 0
    ...

nb_neuron: 4
    nb_weight: 8, 0.72369, -0.140627, -0.388739, ...
    nb_weight_momentum: 8, 0, 0, 0, ...
    bias: -0.393979
    bias_momentum: 0
    ...
```

Now you can use this neural network for whatever you want !

# Prediction mode

In this mode, the program reads a neural network from a file and uses it to predict the outcome of a series of games.  
- The program reads the neural network from the file NEURAL_NETWORK.nn and the games from the file EVALUATION_FILE.  
- The program prints the predicted outcome of each game to the standard output, one per line.  
- The predicted outcome can be: Checkmate White, Checkmate Black, Check White, Check Black, Stalemate, or Nothing.

## Default mode

In this mode, the outcome is directly printed to the standard output without any extra information.

![image](https://github.com/user-attachments/assets/6e7051ed-09e5-46a2-8825-40918f8efa66)

## Color mode

In this mode the outcome is printed with for each prediction the reference result, the predicted result, the weights of the prediction, and 3 colors to indicate which level of precision the prediction is.

Here are the 3 levels of precision:
- Level 1: Only predict Check or Nothing.
- Level 2: Predict Checkmate, Check, Statemate or Nothing.
- Level 3: Predict outcome with color.

![image](https://github.com/user-attachments/assets/58bdd59c-0e4e-4578-997a-86dfea57d166)

# Train mode

In this mode, the program reads a neural network from a file and trains it on the given dataset.  
At the end of the training, the program saves the neural network to the file from which the neural network has been loaded or to a specific file if specified.
In this mode, nothing is printed to the standard output.

# Train with Stat

In this mode, multiple neural networks can be given as arguments. The program will train each of them in parallel on the given dataset and evaluate them on the given evaluation dataset.   
The evaluation result will be used to fine-tune the learning rate of the different neural networks.   
After each epoch, the program will print the current level of progress of the training and the success rate of the neural network prediction on the evaluation set. It will also print the accuracy rate of the neural network and the current learning rate.

![image](https://github.com/user-attachments/assets/47f78a3f-62f8-4106-828d-449ffa9ac2ad)

#### Here is a visual representation of the training process:

![NeuralNetwork (1)](https://github.com/user-attachments/assets/d1935670-ff0a-4334-8bc9-f0ec00e004ee)

# Activation function

We have tested different activation functions to see which one is the best for our neural network. We have found that the best activation function is the Sigmoid function for our specific chess state prediction problem. Because it can output values between 0 and 1 which are easier to interpret. Maybe other activation functions could be better but we chose the sigmoid function for its simplicity and its ability to output values between 0 and 1.   

We have tested the Relu activation function but we have gotten too many dying neurons and the neural network was not able to learn anything. We also to insane values close to infinity which is why we have decided to not use it.

#### Here is a visual representation of the activation functions we have tested:

![395589646-c0d639da-9914-44e1-9b30-c4ecafa1594b](https://github.com/user-attachments/assets/1b671bf1-6551-4a9b-936b-28a007f3d9b2)

# Hyperparameter optimisation
## Learning Rate

We have tested different learning rates optimization algorithms to see which one is the best for our neural network.
The first that we have tested is the constant learning rate. With this approach, the neural network is often stuck in a local minimum, and if not the learning rate is too high for the neural network to have precise results.

The second approach that we have tested is the learning rate decay. With this approach, the learning rate is decreasing over time. This approach is better than the constant learning rate but the learning rate is decreasing too fast or too slow and the neural network is often stuck in a local minimum and learns nearly nothing after a few epochs.

The third approach that we have tested is the learning rate decay with momentum. With this approach, the learning rate is decreasing over time and the momentum is increasing over time. This approach gives us very good results and neural networks learn very fast. The momentum mechanism can get the neural network out of the local minimum and when the global minimum is reached the momentum can stabilize the neural network and decrease the learning rate to get the most precise result.

#### Here is a visual representation of the momentum algorithm:

![6fd62253592b42795c48dc570a17579c](https://github.com/user-attachments/assets/d3b26a57-cf98-4d22-8cf8-6f28f0893d2b)

#### Here is a visual representation of how the momentum algorithm gets out of a local minimum:

![sgd_with_momentum2](https://github.com/user-attachments/assets/e7694504-49ec-4cac-bd72-13b830954cb8)

## Neural Network Layout

To optimize the neural network layout we have tested different layouts with different numbers of layers and different numbers of neurons per layer and we have benchmarked them to see which one is the best.  
The number of layers and the number of neurons per layer are chosen randomly in a pool of possible values which are powers of 2 for the number of neurons per layer and the number of layers is between 1 and 5.   

We have defined the maximum number of neurons per layer as the smallest power of 2, which is greater than or equal to the number of input neurons. The minimum number of neurons per layer is the smallest power of 2 that is greater than or equal to the number of output neurons.

#### Here is the neural network layout with the best success rate that we have found:

![NeuralNetwork](https://github.com/user-attachments/assets/1277e992-eb91-44ad-ab71-f970501c1eb9)

# Training Method
## Splited Dataset

In this method, we split our dataset into two parts. The first part is the training dataset and the second is the evaluation dataset. The training dataset is used to train the neural network and the evaluation dataset is used to evaluate the neural network.   
During the training, after each epoch, the neural network is evaluated on the evaluation dataset and, depending on the evaluation result, some hyperparameters are tweaked.

#### Here is a representation of the split dataset:

![NeuralNetwork (2)](https://github.com/user-attachments/assets/33af3320-1dbc-4fa6-9957-296050933074)

## Cross-Validation

In this method, we split our dataset into multiple parts. At every epoch, we change which part of the dataset we evaluate and train the neural network. This method is a very good way to avoid overfitting.

#### Here is a representation of the cross-validation training method:

![NeuralNetwork (3)](https://github.com/user-attachments/assets/ec34d0fa-d966-409c-8f1b-bec908977f83)

# Benckmark

To define which neural network layout is the best for our problem we have benchmarked different neural network layouts.
Every neural network layout has been trained on the same dataset and evaluated on the same evaluation dataset and they all start with the same learning rate.

#### Here is a classification of the neural network layout we have tested and the success rate of each of them:

![Best Success Rate compared to Network layout](https://github.com/user-attachments/assets/bf7270ab-aaae-4d4c-b4c4-0d1d63f966eb)

We can see that the neural network layout that have the best results is the one with 2 hidden layers containing 256 and 64 neurons.

#### Here is a graph of the success rate of the neural network over epochs:

![Success Rate over Epoch](https://github.com/user-attachments/assets/98251614-ca10-4446-bb5f-377190c8709e)

We can see that at first the neural network learns very fast and then the learning rate decreases and the neural network stabilizes and learns very slowly.
With a bigger dataset, we could achieve even better results.

#### Here is a graph of the learning rate over epochs:

![Learning Rate over Epoch](https://github.com/user-attachments/assets/a5e20a17-6c62-45ad-b3d8-d4d7e789f27e)

We can see that the learning rate oscillates over and over which prevents the neural network from being stuck in a local minimum.
Also, the learning rate decreases over time which allows the neural network to stabilize and have more precise results.

# Dataset Issue

The dataset that we have used is not big enough to have very precise results.
Also, the given dataset that we used for evaluation is not representative of the real repartition of chess game outcomes.

#### Here is a representation of the dataset repartition over position:

![Dataset repartition over position (2)](https://github.com/user-attachments/assets/58fdc183-e0a9-4ad3-b13c-975a1c950d69)

The **AT Dataset** represents the proportion of positions tested to evaluate the neural network by the automated test.  
The **Default Dataset** represents the proportion of positions tested to evaluate the neural network in all previous benchmarks.  
The **Real Dataset** represents the proportion of position randomly generated which represents the real chess game outcome repartition (based on 64,076,489 chess games).

We can see that the dataset that we have used is not representative of the real repetition of chess game outcomes.
Stalemate is way too much represented in the dataset that we have used and the neural network is not able to predict it correctly.

#### Here is a real position repartition:

Position        | Real Dataset | Real Dataset Percentage
--------------- | ------------ | -----------------------
Checkmate White | 2431359      | 3,794 %
Checkmate Black | 2430665      | 3,793 %
Check White     | 17363189     | 27,098 %
Check Black     | 17364745     | 27,100 %
Stalemate       | 1692         | 0,0026 %
Nothing         | 24484839     | 38,212 %
   |   |   
Total           | 64076489     | 100 %

#### Here is the repartition delta between the given dataset and the real dataset:

![Dataset repartition delta over Position](https://github.com/user-attachments/assets/c2133ff9-4435-4ab8-9e1c-40694488b3ec)

Here again we can clearly that the Stalemate is way too much represented in the given dataset, on this graph we can see that it's 221.95 times more represented in the given dataset than in the real dataset.

#### Here is the success rate for each outcome of the neural network on default dataset:

![Default Dataset Success Rate compared to Expected Success Rate (1)](https://github.com/user-attachments/assets/9e05ffe8-364c-45b4-a03a-cc0073860b92)

Here again we can see that the Stalemate is a big issue for the neural network and it's not able to predict it correctly.
