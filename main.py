from NeuralNetwork import NeuralNetwork, ActivationFuncs
import NeuralNetworkTraining
import Plotting
import testing_mini_batch_sizes
from dataloading import load_mnist
import parameters
from FNNattack import attack, make_attacks

testing_mini_batch = False
plotting_accuracy = False
plotting_loss = True
test_attack = True


epochs = 3
minibatch_size = 64
training_limit = 10000
validation_limit = 1000
test_limit = 1000

# Set global parameters
parameters.N_CLASSES = 10
parameters.INPUT_SIZE = 784


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
        input_size=parameters.INPUT_SIZE,
        hidden_size=30,
        output_size=parameters.N_CLASSES,
        activation_func=ActivationFuncs.SIGMOID,  # TODO: relu is not working right?
        learning_rate=32
    )
    
    
    # Initial test configuration.
    

    print("\nStarting training...")
    standard_history = NeuralNetworkTraining.train_network(
        network,
        training_data,
        validation_data,
        epochs=epochs,
        training_limit=training_limit,
        validation_limit=validation_limit,
        minibatch_size=minibatch_size,
    )

    print(f"Testing network on {test_limit:,} test examples...")
    correct, total = NeuralNetworkTraining.evaluate(network, test_data, test_limit)
    accuracy = 100.0 * correct / total

    print()
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy:.2f}%")
    
    


    
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

    if test_attack:
        attack_image_index = 1
        x0 = training_data[0][attack_image_index]
        y0 = training_data[1][attack_image_index]
        make_attacks(network, x0)
