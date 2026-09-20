import pickle
from NeuralNetwork import NeuralNetwork
import NeuralNetworkTraining
import plotting
import testing_mini_batch_sizes


testing_mini_batch = False
plotting_accuracy = False
plotting_loss = True


epochs = 3
training_limit = 10000
validation_limit = 1000
test_limit = 1000

def load_mnist(filename):
    with open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


if __name__ == "__main__":

    dataset_file = "mnist.pkl"

    print("Loading MNIST dataset...")
    training_data, validation_data, test_data = load_mnist(dataset_file)

    print("Dataset loaded.")
    print("Training examples:", len(training_data[0]))
    print("Validation examples:", len(validation_data[0]))
    print("Test examples:", len(test_data[0]))
    print("Inputs per image:", len(training_data[0][0]))

    # --------------------------------------------------------
    # STANDARD RUN
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STANDARD RUN: MINI-BATCH SIZE = 10")
    print("=" * 60)

    network = NeuralNetwork(
        input_size=784,
        hidden_size=30,
        output_size=10,
        learning_rate=3
    )
    
    
    # Initial test configuration.
    

    print("\nStarting training...")
    standard_history = NeuralNetworkTraining.train_network(
        network,
        training_data,
        validation_data,
        epochs=epochs,
        training_limit=training_limit,
        validation_limit=validation_limit
    )

    print(f"Testing network on {test_limit:,} test examples...")
    correct, total = NeuralNetworkTraining.evaluate(network, test_data, test_limit)
    accuracy = 100.0 * correct / total

    print()
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy:.2f}%")# -*- coding: utf-8 -*-
    
    


    
    if testing_mini_batch:
        # --------------------------------------------------------
        # MINI-BATCH EXPERIMENT
        # --------------------------------------------------------

        print("\n" + "=" * 60)
        print("MINI-BATCH SIZE EXPERIMENT")
        print("=" * 60)

        batch_sizes = [1, 10, 20, 50]

        results = testing_mini_batch_sizes.compare_mini_batch_sizes(
            training_data,
            validation_data,
            batch_sizes=batch_sizes,
            epochs=3,
            training_limit=10000,
            validation_limit=1000
        )

        Plotting.plot_batch_accuracy(results)
        Plotting.plot_batch_time(results)
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        for batch_size in batch_sizes:
            print(
                f"Mini-batch {batch_size:>2}: "
                f"validation accuracy = {results[batch_size]['final_accuracy']:.2f}% | "
                f"time = {results[batch_size]['time']:.2f} s"
            )
            

    
    if plotting_accuracy:
        
    # Graphs for standard run
        print("\nCreating accuracy graph...")
        Plotting.plot_validation_accuracy(standard_history)
    
    if plotting_loss:
        print("\nCreating loss graph...")
        Plotting.plot_loss(standard_history)
