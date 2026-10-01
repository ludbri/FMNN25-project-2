"""
Compare the Nerual Network if 10 output nodes are used (i.e 1 per class)
or the 4-bit encoding is used (4 output nodes)
"""

from NeuralNetwork import NeuralNetwork, relu, relu_derivative
import NeuralNetworkTraining
import Plotting
import testing_mini_batch_sizes
from dataloading import load_mnist
import parameters
from FNNattack import attack, make_attacks
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters / limits for the standard training run
epochs = 10
minibatch_size = 32
training_limit = 10000
validation_limit = 1000
test_limit = 1000


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
    # STANDARD RUN REGULAR NETWORK
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"STANDARD RUN: MINI-BATCH SIZE = {minibatch_size}")
    print("=" * 60)

    regular_network = NeuralNetwork(
        layer_sizes=(parameters.INPUT_SIZE,
                     30,
                     parameters.OUTPUT_SIZE),
        # activation_funcs=(relu,) * 2,  # TODO: relu is almost learning.
        # activation_func_gradients=(relu_derivative,) * 2,
        learning_rate=0.3,
        learning_rate_decay=0.1,
        binary_encoding=False
    )
    
    
    print("\nStarting training...")
    rg_standard_history = NeuralNetworkTraining.train_network(
        regular_network,
        training_data,
        validation_data,
        epochs=epochs,
        training_limit=training_limit,
        validation_limit=validation_limit,
        minibatch_size=minibatch_size,
    )

    print(f"Testing regular network on {test_limit:,} test examples...")
    correct, total = NeuralNetworkTraining.evaluate(regular_network, test_data, test_limit)
    accuracy = 100.0 * correct / total

    print(" ----------------- REGULAR NETWORK -----------------")
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy:.2f}%")


    
    # --------------------------------------------------------
    # STANDARD RUN 4-BIT ENCODING
    # --------------------------------------------------------
    

    fourbit_network = NeuralNetwork(
            layer_sizes=(parameters.INPUT_SIZE,
                        30,
                        4),
            # activation_funcs=(relu,) * 2,  # TODO: relu is almost learning.
            # activation_func_gradients=(relu_derivative,) * 2,
            learning_rate=0.3,
            learning_rate_decay=0.1,
            binary_encoding=True
        )
        
        
    print("\nStarting training on 4-bit...")
    fb_standard_history = NeuralNetworkTraining.train_network(
        fourbit_network,
        training_data,
        validation_data,
        epochs=epochs,
        training_limit=training_limit,
        validation_limit=validation_limit,
        minibatch_size=minibatch_size,
    )

    print(f"Testing 4-bit network on {test_limit:,} test examples...")
    correct, total = NeuralNetworkTraining.evaluate(fourbit_network, test_data, test_limit)
    accuracy = 100.0 * correct / total

    print(" ----------------- 4 BIT NETWORK -----------------")
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy:.2f}%")
    
    
    # --------------------------------------------------------
    # MINI-BATCH EXPERIMENT
    # Trains separate networks at several mini-batch sizes to compare
    # their effect on final validation accuracy and training time.
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MINI-BATCH SIZE EXPERIMENT")
    print("=" * 60)

    batch_sizes = [1, 10, 20, 50]

    rg_results = testing_mini_batch_sizes.compare_mini_batch_sizes(
        training_data,
        validation_data,
        batch_sizes=batch_sizes,
        epochs=3,
        training_limit=10000,
        validation_limit=1000,
    )

    fb_results = testing_mini_batch_sizes.compare_mini_batch_sizes(
            training_data,
            validation_data,
            batch_sizes=batch_sizes,
            epochs=3,
            training_limit=10000,
            validation_limit=1000,
        )


    batch_results = {
        "Regular Network": rg_results,
        "4-bit Encoding": fb_results,
    }


    fig, axes = plt.subplots(1, 2, figsize=(12, 10))
    Plotting.plot_batch_accuracy(batch_results, ax=axes[0])
    Plotting.plot_batch_time(batch_results,ax=axes[1])
 

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        print(
            f"Mini-batch {batch_size:>2}: "
            f"validation accuracy = {rg_results[batch_size]['final_accuracy']:.2f}% | "
            f"time = {rg_results[batch_size]['time']:.2f} s"
        )
            

    # ACCURACY GRAPH
    def plot_validation_accuracy_same_graph(history_a, history_b, label_a="A", label_b="B"):
        """
        Plots two validation-accuracy histories on the same axes for direct comparison.
        """
        plt.figure()
        plt.plot(history_a["epochs"], history_a["validation_accuracy"], marker="o", label=label_a)
        plt.plot(history_b["epochs"], history_b["validation_accuracy"], marker="o", label=label_b)
        plt.xlabel("Epoch")
        plt.ylabel("Validation Accuracy (%)")
        plt.title("Validation Accuracy vs Epoch")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("validation_accuracy_comparison.png", dpi=150)
        plt.show()

    plot_validation_accuracy_same_graph(rg_standard_history, fb_standard_history, "Regular Network", "4-bit Encoding")

    # LOSS GRAPH
    def plot_loss_side_by_side(history_a, history_b, label_a="A", label_b="B"):
        """
        Plots training and validation loss for two networks
        side-by-side, each panel showing both loss curves for its network.
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

        for ax, history, label in zip(axes, (history_a, history_b), (label_a, label_b)):
            ax.plot(history["epochs"], history["training_loss"], marker="o", label="training loss")
            ax.plot(history["epochs"], history["validation_loss"], marker="o", label="validation loss")
            ax.set_xlabel("Epoch")
            ax.set_title(label)
            ax.set_xticks(history["epochs"])
            ax.legend()
            ax.grid(True)

        axes[0].set_ylabel("Mean Squared Error")
        fig.suptitle("Training and Validation Loss vs Epoch")
        plt.tight_layout()
        plt.savefig("loss_vs_epoch_side_by_side.png", dpi=150)
        plt.show()

    plot_loss_side_by_side(rg_standard_history, fb_standard_history, "Regular Network", "4-bit encoding")

    # BOTH CONFUSION MATRICES FOR BOTH NETWORKS
    def plot_confusion_grid(network_a, network_b,
                            test_data, validation_data,
                            label_a="A", label_b="B"):
        """
        Plots confusion matrices for two networks on both the test and
        validation sets, as a 2x2 grid.
       """
        def confusion_counts(y_true, y_pred):
            counts = np.zeros((parameters.N_CLASSES, parameters.N_CLASSES), dtype=float)
            for y_t, y_p in zip(y_true, y_pred):
                counts[y_t, y_p] += 1
            counts /= counts.sum(axis=1, keepdims=True)
            return counts

        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        ticks = list(range(parameters.N_CLASSES))

        datasets = (("Test", test_data), ("Validation", validation_data))
        networks = ((label_a, network_a, False), (label_b, network_b, True))

        for row, (dset_name, (images, y_true)) in enumerate(datasets):
            for col, (net_name, network, binary) in enumerate(networks):
                ax = axes[row, col]
                # print("Output Nodes numbers are: " binary, parameters.OUTPUT_SIZE)
                y_pred = network.predict(images)
                counts = confusion_counts(y_true, y_pred)

                ax.matshow(counts, cmap='viridis')
                ax.set_title(f"{net_name} — {dset_name}")
                ax.set_xticks(ticks, ticks)
                ax.set_yticks(ticks, ticks)
                ax.set_xlabel("Predicted")
                if col == 0:
                    ax.set_ylabel("True")

        fig.suptitle("Confusion Matrices")
        plt.tight_layout()
        plt.savefig("confusion_matrix_grid.png", dpi=150)
        plt.show()

        return fig, axes

    plot_confusion_grid(
        regular_network, fourbit_network,
        test_data, validation_data,
        label_a="Regular Network", label_b="4-bit Encoding"
    )

    # Run an adversarial attack against the trained network, using a
    # single training image (at attack_image_index) as the starting point
    attack_image_index = 1
    x0 = training_data[0][attack_image_index]

    fig = plt.figure(figsize=(20, 10))
    subfigs = fig.subfigures(1, 2) 
    make_attacks(regular_network, x0, fig=subfigs[0], title="Regular Network")
    make_attacks(fourbit_network, x0, fig=subfigs[1], title="4-bit Encoding")
    plt.tight_layout()
    plt.savefig("target-fnn-attack.png", dpi=150)
    plt.show()


    fig = plt.figure(figsize=(20, 10))
    subfigs = fig.subfigures(1, 2) 
    make_attacks(regular_network,x0, att="negative_gradient", fig=subfigs[0], title="Regular Network")
    make_attacks(fourbit_network,x0, att="negative_gradient", fig=subfigs[1], title="4-bit Encoding")
    plt.tight_layout()
    plt.savefig("negative-gradient-fnn-attack.png", dpi=150)
    plt.show()