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
    results = {}

    for batch_size in batch_sizes:
        print("=" * 60)
        print(f"MINI-BATCH SIZE = {batch_size}")
        print("=" * 60)

        # Fresh network for a fair separate experiment
        network = NeuralNetwork(
            input_size=parameters.INPUT_SIZE,
            hidden_size=30,
            output_size=parameters.N_CLASSES,
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

        results[batch_size] = {
            "history": history,
            "final_accuracy": history["validation_accuracy"][-1],
            "time": elapsed
        }

        print(f"Final validation accuracy: {history['validation_accuracy'][-1]:.2f}%")
        print(f"Training time: {elapsed:.2f} seconds")
        print()

    return results


