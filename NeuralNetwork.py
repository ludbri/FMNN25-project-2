import numpy as np

from parameters import BINARY_ENCODING
import binary
import warnings

# Source - https://stackoverflow.com/a/1988024
# Posted by Raja Selvaraj, modified by community. See post 'Timeline' for change history
# Retrieved 2026-09-28, License - CC BY-SA 4.0

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)


from loss_activation_and_error_functions import (sigmoid, sigmoid_derivative, 
                                           relu, relu_derivative, 
                                           ActivationFunc, ActivationFuncGrad, 
                                           LossFunc, LossFuncGrad,
                                           square_loss, square_loss_gradient,
                                           _numerical_gradient, _numerical_loss_gradient)


class NeuralNetwork:
    # 3-layer network: INPUT -> HIDDEN -> OUTPUT
    def __init__(self, 
                 layer_sizes: tuple[int],
                 activation_funcs: tuple[ActivationFunc] = None,
                 activation_func_gradients: tuple[ActivationFuncGrad] = None,
                 loss_func: LossFunc = None,
                 loss_func_gradient: LossFuncGrad = None,
                 learning_rate: float = 0.3,
                 learning_rate_decay: float = 0.0):
        """
        Instantiate a feed-forward neural network of the specified dimensions 
        and activation functions.Neuron layers do NOT include a bias term.

        layer_sizes:
            the number of neurons in each layer,
            including the input and output layers.

        activation_funcs:
            the activation functions to use in each layer.
            If a single function is provided, it is used for all non-input layers.
            If None, defaults to sigmoid activation.

        activation_func_gradients:
            Functions for the derivative f'(z) of each layer's activation.
            Each is passed the pre-activation z = W @ x + b, NOT the layer output.
            If activation_funcs is given but this is None, the derivatives are
            computed numerically with central differences.
            If both are None, sigmoid and its derivative are used.

        loss_func:
            the callable loss function to use.
            It is passed the predicted and true label encodings.
            Note, this is not used for training.
            If None, defaults to square error loss.

        loss_func_gradient:
            a callable function for the gradient of the loss function,
            It is passed the predicted and true label encodings.
            Must be provided if loss_func is provided.
            If None, defaults to the gradient of the square error loss.

        learning_rate:
            learning rate divisor. 
        
        learning_rate_decay:
            the parameter defining the decay of the learning rate as epochs 
            increases. The final learning rate = learning_rate/(1 + decay*epoch_count).
            The default value is 0.
        """
        
        
        assert len(layer_sizes) > 2, \
            ValueError(f"The number of layers ({len(layer_sizes)}) " +\
                       " must be >2 to contain an input and output layer.")

        self.layer_sizes = layer_sizes
        self.depth = len(layer_sizes) - 1  # Not counting the input layer
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        # Weights and biases
        # index 0 is the edges before layer 1.
        self.layer_weights = []
        self.biases = []
        
        #?? This should be dependent on which activation function we use, this is for sigmoid
        for i_size, o_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layer_weights.append(np.random.randn(o_size, i_size) / np.sqrt(i_size))
            self.biases.append(np.zeros((o_size, 1)))
        
        '''
        for i_size, o_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layer_weights.append(np.random.uniform(-0.5, 0.5, (o_size, i_size)))
            self.biases.append(np.random.uniform(-0.5, 0.5, (o_size, 1)))'''

        
        # Assigns the activation functions of each layer and their derivatives
        # The default activation is the sigmoid function.
        if activation_funcs is None:
            self.activation_funcs = (sigmoid,) * self.depth
            self.activation_derivatives = (sigmoid_derivative,) * self.depth
        
        else:
            if len(activation_funcs) != self.depth:
                raise ValueError(
                    f"Expected {self.depth} activation functions (one per non-input layer), "
                    f"got {len(activation_funcs)}."
                )
            self.activation_funcs = activation_funcs
        
            if activation_func_gradients is None:
                # no derivatives supplied: fall back to central differences
                self.activation_derivatives = tuple(
                    _numerical_gradient(fn) for fn in activation_funcs
                )
            else:
                if len(activation_func_gradients) != self.depth:
                    raise ValueError(
                        f"Expected {self.depth} activation derivatives (one per non-input layer), "
                        f"got {len(activation_func_gradients)}."
                    )
                self.activation_derivatives = activation_func_gradients

        # Assigns the loss functions and its derivative
        #  The default function is the square error loss function.
        if loss_func is None:
            self.loss_func = square_loss
            self.loss_func_derivative = square_loss_gradient  # This returns positive gradient
        else:
            self.loss_func = loss_func
            if loss_func_gradient is None:
                warnings.warn("No loss gradient given; using slow numerical differentiation.")
                self.loss_func_derivative = _numerical_loss_gradient(self.loss_func)  # This returns positive gradient
            else:
                self.loss_func_derivative = loss_func_gradient  # This returns positive gradient


    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Passes a set of inputs forwards through the network.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (self.input_size, n), where n is the number
            of samples.
    
        Returns
        -------
        layer_outputs : np.ndarray
            Output activations of each layer, including the input layer.
        """
        
        # Ensure x is an ndarray (handles lists/other array-likes passed in)
        x = np.asarray(x)
        
        # Validate that the number of input features matches what the network expects
        if x.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} input size, received {x.shape[0]}.")
        
        # Layer activations:
        # linear transform (weights @ x + bias), followed by activation function
        layer_outputs = [x]
        pre_activations = []  # Store  preactications z = W @ x + b

        for w, b, sigma in zip(self.layer_weights, self.biases, self.activation_funcs):
            z = w @ x + b
            pre_activations.append(z)
            x = sigma(z)
            layer_outputs.append(x)

        return layer_outputs, pre_activations


    # Prediction: output neuron with largest activation
    def predict(self, x: np.ndarray, raw_output=False) -> np.ndarray:
        """
        Passes a set of inputs through the network and returns the predicted
        class labels.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n, input_size), where n is the number
            of samples.

        one_hot: bool
            Flag for if the returned prediction should be the raw network output.
    
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
        if x.shape[1] != self.layer_sizes[0]:
            raise ValueError(f"Expected {self.layer_sizes[0]} input size, received {x.shape[1]}.")
        
        # forward() expects shape (input_size, n), but x here is (n, input_size),
        # so transpose before passing it through; only the output layer
        # activations are needed for prediction, hidden activations are discarded
    
        activations, _ = self.forward(x.T)
        outputs = activations[-1]

        if raw_output:
            return outputs
        else:
            if BINARY_ENCODING:
                return np.array([binary.closest_digit_from_binary(outputs[:, i]) for i in range(outputs.shape[1])])

            else:
                # For each sample (column), return the index of the output node
                # with the highest activation — the predicted class label
                return np.argmax(outputs, axis=0)
        

    def evaluate_loss(self,
                      x: np.ndarray,
                      y_true: np.ndarray):
        """
        Evaluates the network loss on one batch of data.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n, input_size) for n samples.
        y_true : np.ndarray
            One-hot encoded true class labels, of shape (n, output_size).

        Returns
        -------
        float
            The sample average loss, evaluated across the provided samples.
        """
        y_pred = self.predict(x, raw_output=True)
        return self.loss_func(y_pred, y_true.T)


    def learn_batch(self, 
                    x: np.ndarray, 
                    y_true: np.ndarray, 
                    epoch_count: int):
        """
        Trains the network on one batch of data using gradient backpropagation.
    
        Parameters
        ----------
        x : np.ndarray
            Input array of shape (n, input_size) for n samples.
        y_true : np.ndarray
            One-hot encoded true class labels, of shape (n, output_size).
        epoch_count : int
            Current epoch number; used to decay the effective learning rate
            as 1 / (epoch_count*decay + 1).
        """
        decay = self.learning_rate_decay
        # Transpose so each column is a sample: shape becomes (input_size, n) / (output_size, n)
        x = np.asarray(x).reshape(-1, self.layer_sizes[0]).T
        y_true = np.asarray(y_true).reshape(-1, self.layer_sizes[-1]).T
        batch_size = x.shape[1]

        # do the forward pass for the batch
        activations, pre_activations = self.forward(x)

        # positive(!) gradient of loss function
        backgrad = self.loss_func_derivative(activations[-1], y_true)

        # iterate backwards through layers
        # Note, these are stored in reverse order: from last layer to first.
        weight_grads = []
        bias_grads = []

        for prev_act, w, z, actgrad in zip(activations[-2::-1],
                                             self.layer_weights[::-1],
                                             pre_activations[::-1],
                                             self.activation_derivatives[::-1]
                                             ):
            # gradient of output w.r.t. the input of this layer
            a_grad = backgrad * actgrad(z)
            # bias grad is now the mean across samples
            bias_grads.append(
                a_grad.mean(axis=1, keepdims=True)
            )
            # weight grads is the mean across samples
            weight_grads.append(
                (a_grad @ prev_act.T) / batch_size
            )
            backgrad = w.T @ a_grad

        lr = self.learning_rate/(1 + decay*epoch_count)

        for i in range(self.depth):
            self.layer_weights[i] -= lr * weight_grads[-i-1]
            self.biases[i] -= lr * bias_grads[-i-1]


