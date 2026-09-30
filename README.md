# FMNN25-project-2
A project for solving the second project: implementing and training a neural network.

Running the project requires separate downloading and extraction of the mnist.pkl dataset.
The dataset is structured as a tuple of training, validation and test sets. 
Each set is a 2-tuple of a list of image pixel data and a list of corresponding labels.


# File structure

- ``main`` - entrypoint. The main file that is the only runnable file. Uses flags that determine whether to test specific functionality and present particular plots.
- ``parameters`` - defines and sets global parameters.
- ``dataloading`` - reading data from and generating minibatches.
- ``binary`` - functions for encoding and decoding labels as a binary representation.
- ``NeuralNetwork`` - defines the ``NeuralNetwork`` Class, the sigmoid and ReLU activation functions and their gradients, as well as the square loss function and its gradient.
  - The ``NeuralNetwork`` class can be of any depth and layer sizes. Each layer can use a distinct activation function and the user can provide any set of functions per layer. If the gradient is not provided, it is sampled from the function using central differences.
- ``NeuralNetworkTraining`` - functions for training the neural network and evaluating the accuracy of its predictions on some dataset.
- ``FNNattack`` - functions for making a single attack against the network with some target label, and a function for making an attack with each possible target label and plot the results.
- ``Plotting`` - functions for generating plots of training performance.
- ``task8`` - functions for searching over hyperparameters to find low training loss in few epochs.
- ``testing_mini_batch_sizes`` - old version of ``task8``.
- ``AttackDataTraining`` - functions to train model using attacked data.

# Results for task 8
- Using a sigmoid function for the hidden layer and an identity function for the output layer, the loss goes below 10^-8, using the parameters: hidden layers
- The least possible hidden layers obtained with a loss below 10^-8 is ??, with hidden layers=??, learning rate=??, decay=??, batch size ?? and with the sigmoid function for hidden layers and identity function for output layer.

# Contributions
Everyone contributed everywhere with discussions. Areas where individuals worked a bit more:
- Anna: task 8, general debugging and adjustments to reduce loss and increase accuracy, documentation, file organization, binary decoding and encoding
- Annelies: helped initial NN and SGD, intro numpy to reduce time in NN, normalization in SGD, updating to allow general activation functions (updating sigmoid+relu), train model on attacked data + new attack, initial task 8 tests
- Juan: ... 
- Ludwig: minibatcher, numpy-ifying learning functions, generalizing layer sizes and activation functions, first attack function and make_attacks, confusion matrix, documentation
- Tommaso: ... 

# 


# To do:
- [x] Proper documentation
- [ ] Add task results to readme.md
- [x] add a requirements.txt file!

- [ ] Extension
  -  [x] allow any number of hidden layers
  -  [x] compare 4 output neuron with 10 output neuron
  -  [x] allow any activation function to be used
  -  [x] allow any loss to be used
     -  [x] Add example
  -  [x] use attack to generate new data -> train network on this data -> attack network (with new attack)

implementation validation
- [x] Evaluate the losses at the end of each epoch, after learning all minibatches.
  - [x] Plot training and validation losses together.
- [x] Scaling of training gradients? Mean or divide by batchsize? What should it be?
- [x] Normalization of the *total* gradient of the weights and biases.
  - can look at the attack code

- [ ] Task 8 - fine-tuning further (now loss at 10^-3, 0.0019991629169286308)
   - [ ] More for loops (act-funcs, learning rates)!
         - Learning rates are added!
    - [x] Train until loss no longer changes (while loop) + max epoch count
    - [x] We should be able to achieve 10**-8
         - We are able to with 100000 epochs
         - We get loss=9.390e-08, with hidden=30: epochs=2701, lr=0.5, batch=2

- [ ] Presentation? Readme-style.
  - [ ] Who presents what?

Nice to have
- [x] ReLU - leaky relu gradient (small pos gradient for relu=0).
  - [x] Make it work.
- [x] Plotting confusion matrix

We are not aiming to win!
