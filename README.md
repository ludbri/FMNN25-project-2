# FMNN25-project-2
A project for solving the second project: implementing and training a neural network.


#
main - main file that is the only runnable file, contains parameters to determine wether to test 

NeuralNetwork - contains the class for the neural network

NeuralNetworkTraining - contains the functions for training the neural network

Plotting - contains functions for plotting accuracy vs epoch, loss vs epoch and mini-batch size vs training time

testing_mini_batch_sizes - function for testing mini-batch size

To do:
- [x] Proper documentation
- [ ] add a requirements.txt file!

- [ ] Extension
  -  [ ] allow any number of hidden layers
  -  [ ] compare 4 output neuron with 10 output neuron
  -  [ ] allow any activation function to be used
  -  [ ] allow any loss to be used
  -  [ ] use attack to generate new data -> train network on this data -> attack network (with new attack)

implementation validation
- [ ] Evaluate the losses at the end of each epoch, after learning all minibatches.
  - [ ] Plot training and validation losses together.
- [ ] Scaling of training gradients? Mean or divide by batchsize? What should it be?
- [ ] Normalization of the *total* gradient of the weights and biases.
  - can look at the attack code

- [ ] Task 8 - fine-tuning further (now loss at 10^-3, 0.0019991629169286308)
    - More for loops (act-funcs, learning rates)!
    - Train until loss no longer changes (while loop) + max epoch count
    - We should be able to achieve 10**-8

- [ ] Presentation? Readme-style.
  - [ ] Who presents what?

Nice to have
- [ ] ReLU - leaky relu gradient (small pos gradient for relu=0).
- [ ] Plotting confusion matrix

We are not aiming to win!
