import numpy as np
from enum import IntEnum, auto


class ActivationFuncs(IntEnum):
    
    SIGMOID = auto()
    RELU = auto()

# Sigmoid activation function
def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function elementwise to an array:
    sigmoid(x) = 1 / (1 + exp(-x)).

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Array of the same shape as `x`, with the sigmoid function applied
        elementwise. All output values lie in the range (0, 1).
    """

    z = np.clip(x, -700, 700)   # Clip input to avoid overflow in exp(-x);
                                # exp(709) is close to the float64 max,
                                # so values beyond ~±700 would overflow/underflow.
    
    return 1.0 / (1.0 + np.exp(-z)) #applies the sigmoid function

# Derivative of sigmoid
def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Computes the elementwise gradient of the sigmoid activation function,
    using the identity sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).

    Parameters
    ----------
    output : np.ndarray
        The sigmoid-activated output of the layer (i.e., sigmoid(x)),
        not the raw pre-activation input.

    Returns
    -------
    np.ndarray
        Gradient of the sigmoid function, same shape as `output`.
    """

    return x * (1.0 - x) #applies the gradient

def relu(x: np.ndarray) -> np.ndarray:
    """
    Applies the ReLU (Rectified Linear Unit) activation function
    elementwise to an array: relu(x) = max(0, x).
 
    Parameters
    ----------
    x : np.ndarray
        Input array.
 
    Returns
    -------
    np.ndarray
        Array of the same shape as `x`, with ReLU applied elementwise.
    """

    return np.maximum(x,0)

def relu_derivative(output: np.ndarray) -> np.ndarray:
    """
    Computes the elementwise gradient of the ReLU activation function,
    using the ReLU output (relu'(x) = 1 if output > 0 else 0).

    Parameters
    ----------
    output : np.ndarray
        The ReLU-activated output of the layer (i.e., relu(x)),
        not the raw pre-activation input.

    Returns
    -------
    np.ndarray
        Gradient of the ReLU function, same shape as `output`.
    """
    # Note: The numpy.sign function returns -1 if x < 0, 0 if x==0, 1 
    # if x > 0. nan is returned for nan inputs.
    grad = np.sign(output)
    return grad


class NeuralNetwork:
    # 3-layer network: INPUT -> HIDDEN -> OUTPUT
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation_func: ActivationFuncs = ActivationFuncs.SIGMOID,
                 learning_rate: float = 0.3):
        """
        Instantiate a feed-forward neural network of the specified dimensions 
        and activation functions.Neuron layers do NOT include a bias term.

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
        
        # Assigns the activation function and its derivative based on the
        # ActivationFuncs value passed in at construction.
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
    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Passes a set of inputs through the network.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (self.input_size, n), where n is the number
            of samples.
    
        Returns
        -------
        hidden_output : np.ndarray
            Output activations of the hidden layer.
        output : np.ndarray
            Output activations of the output layer.
        """
        
        # Ensure x is an ndarray (handles lists/other array-likes passed in)
        x = np.asarray(x)
        
        # Validate that the number of input features matches what the network expects
        if x.shape[0] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input size, received {x.shape[0]}.")
        
        # Hidden layer: linear transform (weights @ x + bias), then activation
        hidden = self.activation(self.weights_input_hidden @ x + self.bias_hidden)
        
        # Output layer: linear transform of hidden activations, then activation
        output = self.activation(self.weights_hidden_output @ hidden + self.bias_output)
        return hidden, output


    # Prediction: output neuron with largest activation
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Passes a set of inputs through the network and returns the predicted
        class labels.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n, input_size), where n is the number
            of samples.
    
        Returns
        -------
        np.ndarray
            Array of shape (n,) containing, for each sample, the index of
            the output node with the largest activation.
        """
        
        # Ensure x is an ndarray (handles lists/other array-likes passed in)
        x = np.asarray(x)
        
        # If a single sample was passed as a 1D array, promote it to a 2D
        # row vector so the shape checks and transpose below work uniformly
        if x.ndim == 1:
            x = x[np.newaxis,:]
            
        # Validate that the number of input features matches what the network expects    
        if x.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} input size, received {x.shape[1]}.")
        
        # forward() expects shape (input_size, n), but x here is (n, input_size),
        # so transpose before passing it through; only the output layer
        # activations are needed for prediction, hidden activations are discarded
        _, outputs = self.forward(x.T)
        
        # For each sample (column), return the index of the output node
        # with the highest activation — the predicted class label
        return np.argmax(outputs, axis=0)
    
    
    def square_loss(self,
                    y_pred: np.ndarray,
                    y_true: np.ndarray) -> float:
        """
        Calculates the mean squared error between predicted and true values.
    
        Parameters
        ----------
        y_pred : np.ndarray
            Predicted values, of shape (output_size, n) for n samples.
        y_true : np.ndarray
            Ground-truth values, of the same shape as `y_pred`.
    
        Returns
        -------
        float
            The mean squared error, computed as the sum of squared
            differences per sample (averaged over the output dimension via
            np.sum), averaged over all samples, and halved.
        """
        loss = (y_pred - y_true) ** 2
        loss = np.sum(loss, axis=0)
        loss = np.mean(loss) / 2
        return loss
    
    def _normalize(self, grad, eps=1e-8):
        """
        Rescales a gradient array to unit (Frobenius/L2) norm.
    
        Divides `grad` by its norm so the returned array has norm ~1,
        preserving direction but discarding magnitude. Intended to be
        applied separately to each parameter's gradient (e.g. weights
        and biases individually), not to a concatenation of all of them,
        so that one parameter's gradient scale doesn't dominate another's.
    
        Parameters
        ----------
        grad : np.ndarray
            Gradient array to normalize. Can be any shape; the norm is
            computed over all elements.
        eps : float, optional
            Small constant added to the denominator to avoid division
            by zero when `grad` is all zeros (e.g. a dead unit).
            Default is 1e-8.
    
        Returns
        -------
        np.ndarray
            `grad` rescaled to have norm approximately 1, same shape as
            the input. Note this discards the gradient's original
            magnitude entirely -- every call produces a unit-norm step
            regardless of how large or small the true gradient was.
            
        TODO; another option would be to only clip very large gradients??
        """
        norm = np.linalg.norm(grad)
        return grad / (norm + eps)

    # Train one mini-batch using backpropagation
    def train_batch(self, 
                    x: np.ndarray, 
                    y_true: np.ndarray, 
                    epoch_count: int) -> float:
        """
        Trains the network on one batch of data.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n, input_size) for n samples.
        y_true : np.ndarray
            One-hot encoded true class labels, of shape (n, output_size).
        epoch_count : int
            Current epoch number; used to decay the effective learning rate
            as 1 / (epoch_count + 1).
    
        Returns
        -------
        float
            The sample-average loss on the batch, computed AFTER the
            weight update is applied. TODO: evaluate after minibatches.
        """
        # Transpose so each column is a sample: shape becomes (input_size, n) / (output_size, n)
        x = np.asarray(x).reshape(-1, self.input_size).T
        y_true = np.asarray(y_true).reshape(-1, self.output_size).T

        hidden_activation, output_activation = self.forward(x)

        ## Output error and partial gradients
        # loss = (output - y_true) ** 2 = output**2 - 2*output*y_true + y_true**2
        # gradient = 2 * (output - y_true)
        # gradient of loss w.r.t. output layer's output, using the NEGATIVE
        # gradient (y_true - output) so that += lr * grad descends the loss
        output_o_gradient = y_true - output_activation  # Note! Negative gradient!
        # gradient w.r.t. output layer's pre-activation input
        output_i_gradient = output_o_gradient * self.activation_derivative(output_activation)

        # gradient of loss w.r.t. hidden layer's output (backprop through weights)
        hidden_o_gradient = self.weights_hidden_output.T @ output_i_gradient
        # gradient w.r.t. hidden layer's pre-activation input
        hidden_i_gradient = hidden_o_gradient * self.activation_derivative(hidden_activation)



        # Gradients of weights and biases:
        # bias gradients, averaged over the batch
        grad_b_o = output_i_gradient.mean(axis=1, keepdims=True)
        # weight gradients, summed (not averaged) over the batch
        # Q: should this be divided by batch size to match how the bias
        # gradients are averaged above? As-is, larger batches produce
        # proportionally larger weight updates than bias updates. 
        # NO, as we normalize below it is not necessary, but if we decide to not 
        # normalize, we should consider it
        grad_w_ho = (output_i_gradient @ hidden_activation.T)

        # average gradient w.r.t. $b_{T-1}$ - bias terms
        grad_b_h = hidden_i_gradient.mean(axis=1, keepdims=True)
        grad_w_ih = (hidden_i_gradient @ x.T)
        
        # TODO: normalize the gradient to length 1. DONE!

        
        grad_w_ho = self._normalize(grad_w_ho)
        grad_b_o  = self._normalize(grad_b_o)
        grad_w_ih = self._normalize(grad_w_ih)
        grad_b_h  = self._normalize(grad_b_h)

        # Learning rate, decayed by epoch count
        # Q: is this maybe inverted?
        #previously lr = 1 / (epoch_count + 1) / self.learning_rate
        lr = self.learning_rate / (epoch_count + 1)


        # adjust weights
        self.weights_hidden_output += lr * grad_w_ho
        self.bias_output += lr * grad_b_o
        self.weights_input_hidden += lr * grad_w_ih
        self.bias_hidden += lr * grad_b_h
        
        # Mean Squared Error, used here only to monitor training.
        # Recomputes forward() AFTER the weight update above.
        # Q: see the note in Returns above -- is post-update loss intended?
        _, output_activation = self.forward(x)
        batch_loss = self.square_loss(output_activation, y_true)

        return batch_loss
