import pickle
from NeuralNetwork import NeuralNetwork
import NeuralNetworkTraining
import Plotting
import testing_mini_batch_sizes
import numpy as np
import random
from dataloading import load_mnist
import parameters

testing_mini_batch = False
plotting_accuracy = False
plotting_loss = True




training_limit = 10000
validation_limit = 1000
test_limit = 1000


#Changing
hidden_size_tuple = tuple(np.arange(35, 45, 1)) #best 40
epochs_tuple = tuple(np.arange(5, 20, 1)) #best16
#also the activiation function , now sigma
#training proceudre , stochastic gradient descent
mini_batch_size_tuple = (1, 2) #best 1
learning_rate_tuple = tuple(np.arange(3, 5, 0.1)) #best 4.9



def extract_first_x(training_data:tuple, x:int):
    ''' 
    Training_data is a tuple of length 2: first entry is a np.array containing the 
    images, second entry is the number that each image corresponds to. 
    '''
    images, numbers = training_data[0], training_data[1]
    
    training_data_modified = (images[:x], numbers[:x])
    
    return training_data_modified


    

if __name__ == "__main__":

    dataset_file = "mnist.pkl"
    
    random.seed(42)
    
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
    for hidden_size in hidden_size_tuple:
        for learning_rate in learning_rate_tuple:
            for mini_batch_size in mini_batch_size_tuple:
                for epochs in epochs_tuple:
    
                    network = NeuralNetwork(
                        input_size=parameters.INPUT_SIZE,
                        hidden_size=hidden_size,
                        output_size=parameters.N_CLASSES,
                        learning_rate=learning_rate
                    )
        
        
        # Initial test configuration.
        
    
                    standard_history = NeuralNetworkTraining.train_network(
                        network,
                        training_data,
                        validation_data,
                        minibatch_size = mini_batch_size,
                        epochs=epochs,
                        training_limit=training_limit,
                        validation_limit=validation_limit
                    )
                    
                    results = np.append(results, (hidden_size, learning_rate, mini_batch_size, epochs, standard_history["loss"][-1]))
    
    
    print(results[1:].min())
    print(np.where(results == results[1:].min()))
    

