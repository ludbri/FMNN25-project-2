"""
Experiment comparing training outcomes across different mini-batch sizes.
"""

from NeuralNetwork import NeuralNetwork
import time
import NeuralNetworkTraining
import parameters

# ============================================================
# EXPERIMENT: DIFFERENT MINI-BATCH SIZES
# ============================================================

def compare_mini_batch_sizes(training_data, validation_data,
                             batch_sizes, epochs=3,
                             training_limit=10000,
                             validation_limit=1000):
    """
    Trains a separate network for each given mini-batch size and
    compares their final validation accuracy and training time.

    Parameters
    ----------
    training_data : Dataset
        Dataset used for training.
    validation_data : Dataset
        Dataset used to measure validation accuracy after each epoch.
    batch_sizes : list[int]
        Mini-batch sizes to compare; one network is trained per size.
    epochs : int, default=3
        Number of training epochs used for every run.
    training_limit : int, default=10000
        Maximum number of training samples to use per epoch.
    validation_limit : int, default=1000
        Maximum number of validation samples to evaluate on.

    Returns
    -------
    dict
        Dictionary keyed by batch size, where each value is a dict with:
            "history"        : the full HistoryDict returned by train_network
            "final_accuracy" : validation accuracy (%) after the last epoch
            "time"           : wall-clock training time in seconds
    """
    
    results = {}

    for batch_size in batch_sizes:
        print("=" * 60)
        print(f"MINI-BATCH SIZE = {batch_size}")
        print("=" * 60)

        # Fresh network for a fair separate experiment
        network = NeuralNetwork(
            layer_sizes=(parameters.INPUT_SIZE,
                         30,
                         parameters.N_CLASSES),
            learning_rate=0.3
        )

        start_time = time.perf_counter()

        history = NeuralNetworkTraining.train_network(
            network,
            training_data,
            validation_data,
            minibatch_size=batch_size,
            epochs=epochs,
            training_limit=training_limit,
            validation_limit=validation_limit
        )

        elapsed = time.perf_counter() - start_time
        
        # Store full training history plus the two headline metrics
       # used for the summary plots (accuracy vs. batch size, time vs. batch size)
        results[batch_size] = {
            "history": history,
            "final_accuracy": history["validation_accuracy"][-1],
            "time": elapsed
        }

        print(f"Final validation accuracy: {history['validation_accuracy'][-1]:.2f}%")
        print(f"Training time: {elapsed:.2f} seconds")
        print()

    return results


