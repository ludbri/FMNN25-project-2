import gzip
import pickle
import random
import numpy as np


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
        self.weights_input_hidden = np.random.uniform(
            -0.5,0.5, (hidden_size, input_size))

        # Biases: Hidden layer
        self.bias_hidden = np.random.uniform(
            -0.5,0.5, (hidden_size, 1))

        # Weights: Hidden -> Output
        self.weights_hidden_output = np.random.uniform(
            -0.5,0.5, (output_size, hidden_size))

        # Biases: Output layer
        self.bias_output = np.random.uniform(
            -0.5,0.5, (output_size, 1))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        z = np.clip(x, -700, 700)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, output):
        """Derivative of sigmoid when output = sigmoid(x)."""
        return output * (1.0 - output)

    def forward(self, inputs):
        """Calculate hidden and output activations."""
        inputs = np.array(inputs).reshape(-1, 1)
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} inputs, received {len(inputs)}."
            )

        # Input -> Hidden
        total_hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        hidden_outputs = self.sigmoid(total_hidden)

        # Hidden -> Output
        total_output = np.dot(self.weights_hidden_output, hidden_outputs) + self.bias_output
        final_outputs = self.sigmoid(total_output)

        return hidden_outputs, final_outputs

    def predict(self, inputs):
        """Return the output neuron with the highest activation."""
        _, outputs = self.forward(inputs)
        return int(np.argmax(outputs))

    def train(self, mini_batch):
        '''Train the network on a list of (image, target) tuples using SGD and mini-batch'''
        
        batch_size = len(mini_batch)
        
        # Initialize gradient accumulators with zeros
        nabla_w_ih = np.zeros(self.weights_input_hidden.shape)
        nabla_b_h = np.zeros(self.bias_hidden.shape)
        nabla_w_ho = np.zeros(self.weights_hidden_output.shape)
        nabla_b_o = np.zeros(self.bias_output.shape)
        
        # Accumulate gradients of each image in batch
        for inputs, targets in mini_batch:
            x = np.array(inputs).reshape(-1, 1)
            y = np.array(targets).reshape(-1, 1)
            hidden_outputs, final_outputs = self.forward(inputs)
            
            # output layer error and gradients
            output_errors = y - final_outputs # error
            
            output_gradients = output_errors * self.sigmoid_derivative(final_outputs) # gradient
            
            # hidden layer error and gradients
            hidden_errors = np.dot(self.weights_hidden_output.T, output_gradients) # error
                
            hidden_gradients = hidden_errors * self.sigmoid_derivative(hidden_outputs) # gradient
            
            # add image gradient to accumalators
            nabla_b_o += output_gradients
            nabla_w_ho += np.dot(output_gradients, hidden_outputs.T)

            nabla_b_h += hidden_gradients
            nabla_w_ih += np.dot(hidden_gradients, x.T)
        
        # apply average updates to weights
        effective_lr = self.learning_rate / batch_size
        
        self.weights_hidden_output += effective_lr * nabla_w_ho
        self.bias_output += effective_lr * nabla_b_o

        self.weights_input_hidden += effective_lr * nabla_w_ih
        self.bias_hidden += effective_lr * nabla_b_h

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


def train_network(network, training_data, validation_data, mini_batch_size = 10,
                  epochs=3, training_limit=10000, validation_limit=1000):
    """Train on a subset and report validation accuracy after each epoch."""
    images, labels = training_data
    training_limit = min(training_limit, len(images))
    validation_limit = min(validation_limit, len(validation_data[0]))

    indices = list(range(training_limit))

    for epoch in range(epochs):
        random.shuffle(indices)
        
        # slice shuffled indices into minibatch
        mini_batches = [
            indices[k : k + mini_batch_size]
            for k in range(0, training_limit, mini_batch_size)
        ]

        for count, batch_indices in enumerate(mini_batches, start=1):
            
            # transform raw data into tuples for network
            mini_batch = [
                (images[idx], one_hot(int(labels[idx])))
                for idx in batch_indices
            ]
            
            # send chunk to be processed at once
            network.train(mini_batch)

            if count % max(1, (500 // mini_batch_size)) == 0:
                processed_examples = min(count * mini_batch_size, training_limit)
                print(
                    f"  Epoch {epoch + 1}/{epochs}: "
                    f"Processed {processed_examples}/{training_limit} examples"
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
