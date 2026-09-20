import gzip
import pickle
import random
import time
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    # 3-layer network: INPUT -> HIDDEN -> OUTPUT
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.c = 1 / learning_rate

        # Weights and biases
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.bias_output = np.random.uniform(-0.5, 0.5, (output_size, 1))

    # Sigmoid activation function
    def sigmoid(self, x):
        z = np.clip(x, -700, 700)
        return 1.0 / (1.0 + np.exp(-z))

    # Derivative of sigmoid
    def sigmoid_derivative(self, output):
        return output * (1.0 - output)

    # Forward propagation
    def forward(self, inputs):
        x = np.asarray(inputs).reshape(-1, 1)
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, received {x.shape[0]}.")

        hidden = self.sigmoid(self.weights_input_hidden @ x + self.bias_hidden)
        output = self.sigmoid(self.weights_hidden_output @ hidden + self.bias_output)
        return hidden, output

    # Prediction: output neuron with largest activation
    def predict(self, inputs):
        _, outputs = self.forward(inputs)
        return int(np.argmax(outputs))

    # Train one mini-batch using backpropagation
    def train(self, mini_batch, j):
        batch_size = len(mini_batch)

        grad_w_ih = np.zeros_like(self.weights_input_hidden)
        grad_b_h = np.zeros_like(self.bias_hidden)
        grad_w_ho = np.zeros_like(self.weights_hidden_output)
        grad_b_o = np.zeros_like(self.bias_output)

        batch_loss = 0.0

        for inputs, targets in mini_batch:
            x = np.asarray(inputs).reshape(-1, 1)
            y = np.asarray(targets).reshape(-1, 1)

            hidden, output = self.forward(x)

            # Mean Squared Error, used here only to monitor training
            batch_loss += np.mean((y - output) ** 2)

            # Output error and gradient
            output_error = y - output
            output_gradient = output_error * self.sigmoid_derivative(output)

            # Hidden error and gradient
            hidden_error = self.weights_hidden_output.T @ output_gradient
            hidden_gradient = hidden_error * self.sigmoid_derivative(hidden)

            # Accumulate gradients
            grad_b_o += output_gradient
            grad_w_ho += output_gradient @ hidden.T
            grad_b_h += hidden_gradient
            grad_w_ih += hidden_gradient @ x.T

        # Average gradient over the mini-batch
        lr = 1/ (self.c * (j + 1) * batch_size)

        self.weights_hidden_output += lr * grad_w_ho
        self.bias_output += lr * grad_b_o
        self.weights_input_hidden += lr * grad_w_ih
        self.bias_hidden += lr * grad_b_h

        return batch_loss / batch_size


def load_mnist(filename):
    with open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


def one_hot(label):
    target = np.zeros(10)
    target[int(label)] = 1.0
    return target


def evaluate(network, dataset, limit=None):
    images, labels = dataset
    if limit is None:
        limit = len(images)
    else:
        limit = min(limit, len(images))

    correct = 0
    for i in range(limit):
        if network.predict(images[i]) == int(labels[i]):
            correct += 1

    return correct, limit


def train_network(network, training_data, validation_data,
                  mini_batch_size=10, epochs=3,
                  training_limit=10000, validation_limit=1000):
    # Save metrics for plots
    images, labels = training_data
    training_limit = min(training_limit, len(images))
    validation_limit = min(validation_limit, len(validation_data[0]))

    indices = list(range(training_limit))

    history = {
        "epochs": [],
        "loss": [],
        "validation_accuracy": []
    }
    global_step = 0

    for epoch in range(epochs):
        random.shuffle(indices)

        total_loss = 0.0
        batches = 0

        for start in range(0, training_limit, mini_batch_size):
            batch_indices = indices[start:start + mini_batch_size]

            mini_batch = [
                (images[i], one_hot(labels[i]))
                for i in batch_indices
            ]

            total_loss += network.train(mini_batch, j = epoch) # j resets after each epoch, is that what we want?
            global_step += 1
            batches += 1

            processed = min(start + mini_batch_size, training_limit)
            if processed % 1000 == 0:
                print(f"  Epoch {epoch + 1}/{epochs}: {processed}/{training_limit} examples")

        average_loss = total_loss / batches

        correct, total = evaluate(
            network,
            validation_data,
            validation_limit
        )
        validation_accuracy = 100.0 * correct / total

        history["epochs"].append(epoch + 1)
        history["loss"].append(average_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} completed.")
        print(f"  Loss: {average_loss:.6f}")
        print(f"  Validation accuracy: {validation_accuracy:.2f}%")
        print()

    return history


# ============================================================
# GRAPH 1: VALIDATION ACCURACY VS EPOCH
# ============================================================

def plot_validation_accuracy(history):
    plt.figure()
    plt.plot(history["epochs"], history["validation_accuracy"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title("Validation Accuracy vs Epoch")
    plt.xticks(history["epochs"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_accuracy_vs_epoch.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 2: LOSS VS EPOCH
# ============================================================

def plot_loss(history):
    plt.figure()
    plt.plot(history["epochs"], history["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.title("Training Loss vs Epoch")
    plt.xticks(history["epochs"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_vs_epoch.png", dpi=150)
    plt.show()


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
            input_size=784,
            hidden_size=30,
            output_size=10,
            learning_rate=0.3
        )

        start_time = time.perf_counter()

        history = train_network(
            network,
            training_data,
            validation_data,
            mini_batch_size=batch_size,
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


# ============================================================
# GRAPH 3: MINI-BATCH SIZE VS VALIDATION ACCURACY
# ============================================================

def plot_batch_accuracy(results):
    batch_sizes = list(results.keys())
    accuracies = [results[b]["final_accuracy"] for b in batch_sizes]

    plt.figure()
    plt.plot(batch_sizes, accuracies, marker="o")
    plt.xlabel("Mini-batch size")
    plt.ylabel("Final Validation Accuracy (%)")
    plt.title("Mini-batch Size vs Validation Accuracy")
    plt.xticks(batch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mini_batch_vs_accuracy.png", dpi=150)
    plt.show()


# ============================================================
# GRAPH 4: MINI-BATCH SIZE VS TRAINING TIME
# ============================================================

def plot_batch_time(results):
    batch_sizes = list(results.keys())
    times = [results[b]["time"] for b in batch_sizes]

    plt.figure()
    plt.plot(batch_sizes, times, marker="o")
    plt.xlabel("Mini-batch size")
    plt.ylabel("Training Time (seconds)")
    plt.title("Mini-batch Size vs Training Time")
    plt.xticks(batch_sizes)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mini_batch_vs_time.png", dpi=150)
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset_file = "mnist.pkl.gz"

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
        learning_rate=0.3
    )

    standard_history = train_network(
        network,
        training_data,
        validation_data,
        mini_batch_size=10,
        epochs=3,
        training_limit=10000,
        validation_limit=1000
    )

    # Final test
    correct, total = evaluate(network, test_data, 1000)
    print(f"Test accuracy: {100.0 * correct / total:.2f}%")

    # Graphs for standard run
    print("\nCreating accuracy and loss graphs...")
    plot_validation_accuracy(standard_history)
    plot_loss(standard_history)

    # --------------------------------------------------------
    # MINI-BATCH EXPERIMENT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MINI-BATCH SIZE EXPERIMENT")
    print("=" * 60)

    batch_sizes = [1, 10, 20, 50]

    results = compare_mini_batch_sizes(
        training_data,
        validation_data,
        batch_sizes=batch_sizes,
        epochs=3,
        training_limit=10000,
        validation_limit=1000
    )

    plot_batch_accuracy(results)
    plot_batch_time(results)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for batch_size in batch_sizes:
        print(
            f"Mini-batch {batch_size:>2}: "
            f"validation accuracy = {results[batch_size]['final_accuracy']:.2f}% | "
            f"time = {results[batch_size]['time']:.2f} s"
        )

    print("\nSaved graphs:")
    print("  validation_accuracy_vs_epoch.png")
    print("  training_loss_vs_epoch.png")
    print("  mini_batch_vs_accuracy.png")
    print("  mini_batch_vs_time.png")
