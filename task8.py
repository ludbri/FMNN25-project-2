"""
Hyperparameter grid search for NeuralNetwork over hidden layer size,
learning rate, mini-batch size, and epoch count.

Trains a network for every combination of the four parameter tuples
below (nested loop = full grid search), recording the final training
loss for each combination in `results`, then reports the combination
with the lowest final loss.

"""

from NeuralNetwork import NeuralNetwork
import NeuralNetworkTraining
import numpy as np
import random
from dataloading import load_mnist
import parameters


training_limit = 10000
validation_limit = 1000
test_limit = 1000



# Hyperparameter search ranges. Comments note the best value found in a
# previous run of this search (kept for reference).
hidden_size_tuple = tuple(np.arange(35, 45, 1)) #best 40
epochs_tuple = tuple(np.arange(5, 20, 1)) #best 16
mini_batch_size_tuple = (1, 2) #best 1
learning_rate_tuple = tuple(np.arange(3, 5, 0.1)) #best 4.9

# also the activation function, now sigma, add here for activation option
# training procedure, stochastic gradient descent



def extract_first_x(training_data: tuple, 
                    x: int) -> tuple:
    '''
    Returns the first `x` samples of a training dataset.

    Parameters
    ----------
    training_data : tuple
        A tuple of length 2: the first entry is an np.array of images,
        the second entry is the array of true labels each image
        corresponds to.
    x : int
        Number of samples to keep, taken from the start of the dataset.

    Returns
    -------
    tuple
        A (images, numbers) tuple truncated to the first `x` samples.
    '''
    images, numbers = training_data[0], training_data[1]
    
    training_data_modified = (images[:x], numbers[:x])
    
    return training_data_modified


    

if __name__ == "__main__":

    dataset_file = "mnist.pkl"
    
    np.random.seed(42)
    
    print("Loading MNIST dataset...")
    training_data, validation_data, test_data = load_mnist(dataset_file)
    training_data = (extract_first_x(training_data, 50))


    print("Dataset loaded.")
    print("Training examples:", len(training_data[0]))
    print("Validation examples:", len(validation_data[0]))
    print("Test examples:", len(test_data[0]))
    print("Inputs per image:", len(training_data[0][0]))
    
    results = np.array(tuple)

     # --------------------------------------------------------
    # STANDARD RUN
    # --------------------------------------------------------
    # Grid search: train a fresh network for every combination of
    # hidden_size, learning_rate, mini_batch_size, and epochs.
    for hidden_size in hidden_size_tuple:
        for learning_rate in learning_rate_tuple:
            for mini_batch_size in mini_batch_size_tuple:
                for epochs in epochs_tuple:
                    
                    #add activation_function thing here also
                    network = NeuralNetwork(
                        layer_sizes=(parameters.INPUT_SIZE,
                                     hidden_size,
                                     parameters.N_CLASSES),
                        learning_rate=learning_rate
                    )
        
        
                    # Initial test configurationn.
        
    
                    standard_history = NeuralNetworkTraining.train_network(
                        network,
                        training_data,
                        validation_data,
                        minibatch_size = mini_batch_size,
                        epochs=epochs,
                        training_limit=training_limit,
                        validation_limit=validation_limit
                    )
                    
                    # Record this combination's hyperparameters alongside
                    # its final training loss, as a single tuple.
                    results.append((hidden_size, learning_rate, mini_batch_size, epochs, standard_history["training_loss"][-1]))

    
    
    print(results[1:].min())
    print(np.where(results == results[1:].min()))
    

