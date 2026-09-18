import gzip
import pickle
import math
import random


class NeuralNetwork:
    """
    Feedforward Neural Network with 3 layers:

        INPUT -> HIDDEN -> OUTPUT

    Uses sigmoid activation and backpropagation.
    No TensorFlow or PyTorch are used.
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=3.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights: Input -> Hidden
        self.weights_input_hidden = [
            [random.uniform(-0.5, 0.5) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]

        # Biases: Hidden layer
        self.bias_hidden = [
            random.uniform(-0.5, 0.5) for _ in range(hidden_size)
        ]

        # Weights: Hidden -> Output
        self.weights_hidden_output = [
            [random.uniform(-0.5, 0.5) for _ in range(hidden_size)]
            for _ in range(output_size)
        ]

        # Biases: Output layer
        self.bias_output = [
            random.uniform(-0.5, 0.5) for _ in range(output_size)
        ]

    def sigmoid(self, x):
        """Sigmoid activation function."""
        if x < -700:
            return 0.0
        if x > 700:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def sigmoid_derivative(self, output):
        """Derivative of sigmoid when output = sigmoid(x)."""
        return output * (1.0 - output)

    def forward(self, inputs):
        """Calculate hidden and output activations."""
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs, received {len(inputs)}."
            )

        # Input -> Hidden
        hidden_outputs = []
        for h in range(self.hidden_size):
            total = self.bias_hidden[h]
            for i in range(self.input_size):
                total += self.weights_input_hidden[h][i] * inputs[i]
            hidden_outputs.append(self.sigmoid(total))

        # Hidden -> Output
        final_outputs = []
        for o in range(self.output_size):
            total = self.bias_output[o]
            for h in range(self.hidden_size):
                total += self.weights_hidden_output[o][h] * hidden_outputs[h]
            final_outputs.append(self.sigmoid(total))

        return hidden_outputs, final_outputs

    def predict(self, inputs):
        """Return the output neuron with the highest activation."""
        _, outputs = self.forward(inputs)
        return outputs.index(max(outputs))

    def train(self, inputs, targets):
        """Train the network on one example using backpropagation."""
        hidden_outputs, final_outputs = self.forward(inputs)

        # Output layer error
        output_errors = [
            targets[o] - final_outputs[o]
            for o in range(self.output_size)
        ]

        # Output layer gradients
        output_gradients = [
            output_errors[o] * self.sigmoid_derivative(final_outputs[o])
            for o in range(self.output_size)
        ]

        # Propagate error to hidden layer
        hidden_errors = []
        for h in range(self.hidden_size):
            error = 0.0
            for o in range(self.output_size):
                error += (
                    self.weights_hidden_output[o][h]
                    * output_gradients[o]
                )
            hidden_errors.append(error)

        # Hidden layer gradients
        hidden_gradients = [
            hidden_errors[h] * self.sigmoid_derivative(hidden_outputs[h])
            for h in range(self.hidden_size)
        ]

        # Update Hidden -> Output weights
        for o in range(self.output_size):
            for h in range(self.hidden_size):
                self.weights_hidden_output[o][h] += (
                    self.learning_rate
                    * output_gradients[o]
                    * hidden_outputs[h]
                )

        # Update output biases
        for o in range(self.output_size):
            self.bias_output[o] += self.learning_rate * output_gradients[o]

        # Update Input -> Hidden weights
        for h in range(self.hidden_size):
            for i in range(self.input_size):
                self.weights_input_hidden[h][i] += (
                    self.learning_rate
                    * hidden_gradients[h]
                    * inputs[i]
                )

        # Update hidden biases
        for h in range(self.hidden_size):
            self.bias_hidden[h] += self.learning_rate * hidden_gradients[h]


# def load_mnist(filename):
#     """
#     Load the supplied mnist.pkl.gz file.
#     Each split has the form: (images, labels).
#     """
#     with gzip.open(filename, "rb") as f:
#         training_data, validation_data, test_data = pickle.load(
#             f, encoding="latin1"
#         )
#     return training_data, validation_data, test_data

def load_mnist(filename):
    with open(filename, "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return training_data, validation_data, test_data


def one_hot(label):
    """Convert digit 0..9 to a 10-element one-hot vector."""
    target = [0.0] * 10
    target[int(label)] = 1.0
    return target


def evaluate(network, dataset, limit=None):
    """Return (number_correct, number_tested)."""
    images, labels = dataset
    if limit is None:
        limit = len(images)
    else:
        limit = min(limit, len(images))

    correct = 0
    for index in range(limit):
        image = images[index]
        label = int(labels[index])
        if network.predict(image) == label:
            correct += 1

    return correct, limit


def train_network(network, training_data, validation_data,
                  epochs=3, training_limit=10000, validation_limit=1000):
    """Train on a subset and report validation accuracy after each epoch."""
    images, labels = training_data
    training_limit = min(training_limit, len(images))
    validation_limit = min(validation_limit, len(validation_data[0]))

    indices = list(range(training_limit))

    for epoch in range(epochs):
        random.shuffle(indices)

        for count, index in enumerate(indices, start=1):
            image = images[index]
            label = int(labels[index])
            network.train(image, one_hot(label))

            if count % 500 == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"{count}/{training_limit} examples"
                )

        print(f"Epoch {epoch + 1}/{epochs} completed.")

        correct, total = evaluate(
            network, validation_data, validation_limit
        )
        validation_accuracy = 100.0 * correct / total
        print(
            f"Validation: {correct}/{total} "
            f"({validation_accuracy:.2f}%)"
        )
        print()


if __name__ == "__main__":
    dataset_file = "mnist.pkl.gz"

    print("Loading MNIST dataset...")
    training_data, validation_data, test_data = load_mnist(dataset_file)

    print("Dataset loaded.")
    print("Training examples:", len(training_data[0]))
    print("Validation examples:", len(validation_data[0]))
    print("Test examples:", len(test_data[0]))
    print("Inputs per image:", len(training_data[0][0]))

    # 784 inputs = 28 x 28 pixels
    # 100 hidden neurons
    # 10 outputs = digits 0..9
    network = NeuralNetwork(
        input_size=784,
        hidden_size=30,
        output_size=10,
        learning_rate=3.0
    )

    # Initial test configuration.
    epochs = 3
    training_limit = 10000
    validation_limit = 1000
    test_limit = 1000

    print("\nStarting training...")
    train_network(
        network,
        training_data,
        validation_data,
        epochs=epochs,
        training_limit=training_limit,
        validation_limit=validation_limit
    )

    print(f"Testing network on {test_limit:,} test examples...")
    correct, total = evaluate(network, test_data, test_limit)
    accuracy = 100.0 * correct / total

    print()
    print("Correct:", correct)
    print("Total:", total)
    print(f"Accuracy: {accuracy:.2f}%")
