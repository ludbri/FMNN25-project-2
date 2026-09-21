import numpy as np
from enum import IntEnum, auto


class ActivationFuncs(IntEnum):
    SIGMOID = auto()
    RELU = auto()

# Sigmoid activation function
def sigmoid(x: np.array) -> np.array:
    """
    Sigmoid activation function, element-by-element.

    x:
        input matrix

    return:
        output activations 1/(1+exp(-x)) of same dimension as the input
    """
    z = np.clip(x, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(output: np.array) -> np.array:
    """
    Gradient of sigmoid activation function, element-by-element.

    output:
        output matrix of the layer

    return:
        gradient of output activations of same dimension as the input
    """
    return output * (1.0 - output)

def relu(x: np.array) -> np.array:
    """
    ReLU (Rectified Linear Unit) activation function, element-by-element.

    x:
        input matrix

    return:
        output activations max(0,x) of same dimension as the input
    """
    return np.maximum(x,0)

def relu_derivative(output: np.array) -> np.array:
    """
    Gradient of ReLU (Rectified Linear Unit) activation function, element-by-element.

    output:
        output matrix of the layer

    return:
        gradient of output activations of same dimension as the input
    """
    # Note: The numpy.sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
    return np.sign(output)


class NeuralNetwork:
    # 3-layer network: INPUT -> HIDDEN -> OUTPUT
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation_func: ActivationFuncs = ActivationFuncs.SIGMOID,
                 learning_rate: float = 0.3):
        """
        Instantiate a feed-forward neural network of the specified dimensions and activation functions.
        Neuron layers do NOT include a bias term.

        input_size:
            the number of neurons in the input layer.

        hidden_size:
            the number of neurons in the hidden layer.

        output_size:
            the number of neurons in the output layer.

        activation_func:
            the activation function to use, assumed same for all layers.

        learning_rate:
            learning rate divisor. TODO: more on this.
        """
        # TODO: arbitrary layer counts
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weights and biases
        self.weights_input_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        self.weights_hidden_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.bias_output = np.random.uniform(-0.5, 0.5, (output_size, 1))

        # Activation functions
        match activation_func:
            case ActivationFuncs.SIGMOID:
                self.activation = sigmoid
                self.activation_derivative = sigmoid_derivative

            case ActivationFuncs.RELU:
                self.activation = relu
                self.activation_derivative = relu_derivative

            case _:
                raise ValueError(f"Activation function must be one of ActivationFuncs, received {activation_func}")

    # Forward propagation
    def forward(self, x: np.array) -> tuple[np.array, np.array]:
        """
        Passes a set of inputs through the network.
        
        x:
            is a input_size x n matrix of n samples.

        return:
            the output activations of the hidden and output layer neurons
        """
        x = np.asarray(x)
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input size, received {x.shape[0]}.")

        hidden = self.activation(self.weights_input_hidden @ x + self.bias_hidden)
        output = self.activation(self.weights_hidden_output @ hidden + self.bias_output)
        return hidden, output


    # Prediction: output neuron with largest activation
    def predict(self, x: np.array) -> np.array:
        """
        Passes a set of inputs through the network returning the predicted classifications.
        
        x:
            is a n x input_size matrix of n samples.

        return:
            a n array with indices of the output node with the largest activation for each input sample.
        """
        x = np.asarray(x)
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input size, received {x.shape[1]}.")
        
        _, outputs = self.forward(x.T)
        return np.argmax(outputs, axis=0)
    
    
    def square_loss(self,
                    y_pred: np.array,
                    y_true: np.array) -> float:
        """
        Calculates the quadratic loss function, taking in the predicted values and 
        the exact values as arrays. 

        the inputs are output_size x n matrices of n samples.
        
        Returns the L2-norm
        """
        loss = (y_pred - y_true) ** 2
        loss = np.sum(loss, axis=0)
        loss = np.mean(loss) / 2
        return loss


    # Train one mini-batch using backpropagation
    def train_batch(self, 
                    x: np.array, 
                    y_true: np.array, 
                    learning_rate: float) -> float:
        """
        Trains the network on one batch of data.

        x:
            a n x input_size matrix of n samples.

        y:
            a n x output_size matrix of the one-hot encoded true classes of each sample.

        learning_rate:
            external learning rate, is divided by the internal self.learning_rate coefficient.

        Returns the sample-average loss on the batch before adjusting weights.
        """
        # Transpose s.t. each column is a sample
        x = np.asarray(x).reshape(-1, self.input_size).T
        y_true = np.asarray(y_true).reshape(-1, self.output_size).T

        hidden_activation, output_activation = self.forward(x)

        # Mean Squared Error, used here only to monitor training
        batch_loss = self.square_loss(output_activation, y_true)

        ## Output error and partial gradients
        # loss = (output - y_true) ** 2 = output**2 - 2*output*y_true + y_true**2
        # gradient = 2 * (output - y_true)
        # gradient of loss w.r.t. $o_T$  (output of final layer)
        output_o_gradient = y_true - output_activation  # Note! Negative gradient!
        # gradient w.r.t. $a_T$  (input of final layer)
        output_i_gradient = output_o_gradient * self.activation_derivative(output_activation)
        # Hidden error and gradient
        # gradient of loss w.r.t. $o_{T-1}$  (output of hidden layer)
        hidden_o_gradient = self.weights_hidden_output.T @ output_i_gradient
        # gradient of loss w.r.t. $a_T$   (input of hidden layer)
        hidden_i_gradient = hidden_o_gradient * self.activation_derivative(hidden_activation)

        # # Gradients of weights and biases:
        # average gradient w.r.t. $b_T$ - bias terms
        grad_b_o = output_i_gradient.mean(axis=1, keepdims=True)
        # average gradient w.r.t. $w_T$ - weight terms
        grad_w_ho = (output_i_gradient @ hidden_activation.T)

        # average gradient w.r.t. $b_{T-1}$ - bias terms
        grad_b_h = hidden_i_gradient.mean(axis=1, keepdims=True)
        # average gradient w.r.t. $w_{T-1}$ - weight terms
        grad_w_ih = (hidden_i_gradient @ x.T)

        # Learning rate
        lr = learning_rate / self.learning_rate

        # adjust weights
        self.weights_hidden_output += lr * grad_w_ho
        self.bias_output += lr * grad_b_o
        self.weights_input_hidden += lr * grad_w_ih
        self.bias_hidden += lr * grad_b_h

        return batch_loss
