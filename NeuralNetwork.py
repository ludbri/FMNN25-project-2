import numpy as np
from enum import IntEnum, auto
from typing import Callable


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
def sigmoid_derivative(input: np.ndarray) -> np.ndarray:
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

    # return output * (1.0 - output) #applies the gradient
    s = sigmoid(input)
    return s * (1.0 - s) # follows numerical gradient and uses input rather than output

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

def relu_derivative(input: np.ndarray) -> np.ndarray:
    """
    Computes the elementwise gradient of the ReLU activation function,
    using the ReLU output (relu'(x) = 1 if output > 0 else 0).
    Leaky ReLU, where negative values result in a small positive gradient.

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
    # grad = np.sign(output)
    # # add a small positive gradient for negative outputs.
    # grad += 10**-3
    # return grad
    
    return np.where(input > 0, 1.0, 0.001)


def square_loss(y_pred: np.ndarray,
                y_true: np.ndarray) -> float:
    """
    Calculates the halved mean squared error between predicted and true encoded values.

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

def square_loss_gradient(y_pred: np.ndarray,
                         y_true: np.ndarray) -> np.ndarray:
    """
    Calculates the gradient of the mean squared error between predicted and true values.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, of shape (output_size, n) for n samples.
    y_true : np.ndarray
        Ground-truth values, of the same shape as `y_pred`.

    Returns
    -------
    np.ndarray
        The (positive) gradient of the mean squared error for each sample. TODO: should this be averaged across samples?
    """
    return y_pred - y_true



# An activation function evaluates a matrix of inputs element-by-element.
ActivationFunc = Callable[[np.ndarray], np.ndarray]
# An activation function gradient evaluates the gradient for a matrix of layer outputs, element-by-element.
ActivationFuncGrad = Callable[[np.ndarray], np.ndarray]
# A loss function takes matrices of the predicted and true label encodings and returns a float
LossFunc = Callable[[np.ndarray,np.ndarray], float]
# A loss function gradient takes matrices of predicted and true label encodings
#   and returns a vector of the gradient with respect to the network output
LossFuncGrad = Callable[[np.ndarray,np.ndarray], np.ndarray]


def _numerical_gradient(func: ActivationFunc, eps: float = 1e-06) -> ActivationFuncGrad:
    
    '''
    Computes the elementwise derivative of an activation function
    using the central difference formula: f'(x) ≈ (f(x + h) - f(x - h)) / (2 * h). (2nd order)
    
    Parameters
    ----------
    func : ActivationFunc
        The activation function f(x).
    eps : float, optional
        Step size for numerical differentiation. Default is 1e-6 for float64 precision.

    Returns
    -------
    ActivationFuncGrad
        A callable function that takes pre-activation inputs `x` 
        and returns the elementwise numerical derivative.
    '''

    # TODO: numerical gradient <- Not sure is there is a way to do with output, adjusted given functions to follow input structure
    def func_grad(input: np.ndarray) -> np.ndarray:
        return (func(input + eps) - func(input - eps)) / (2.0 * eps)
    return func_grad

def _numerical_loss_gradient(func: LossFunc, eps: float = 1e-6) -> LossFuncGrad:
    """
    Numerically computes the gradient of a loss function with respect to y_pred
    using element-wise central difference.
    """
    def func_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(y_pred)
        rows, cols = y_pred.shape
        
        # Perturb each element in y_pred individually
        for i in range(rows):
            for j in range(cols):
                orig_val = y_pred[i, j]
                
                # Perturb +eps
                y_pred[i, j] = orig_val + eps
                loss_plus = func(y_pred, y_true)
                
                # Perturb -eps
                y_pred[i, j] = orig_val - eps
                loss_minus = func(y_pred, y_true)
                
                # Restore original value
                y_pred[i, j] = orig_val
                
                # Central difference derivative
                grad[i, j] = (loss_plus - loss_minus) / (2.0 * eps)
                
        # Scaling adjustment: square_loss computes the AVERAGE loss across batch size N (via np.mean)
        # Because learn_batch divides by batch_size again during weight update,
        # multiplying by batch_size aligns the numerical gradient scale with square_loss_gradient.
        batch_size = y_pred.shape[1]
        return grad * batch_size

    return func_grad



class NeuralNetwork:
    # 3-layer network: INPUT -> HIDDEN -> OUTPUT
    def __init__(self, 
                 layer_sizes: tuple[int],
                 activation_funcs: tuple[ActivationFunc] = None,
                 activation_func_gradients: tuple[ActivationFuncGrad] = None,
                 loss_func: LossFunc = None,
                 loss_func_gradient: LossFuncGrad = None,
                 learning_rate: float = 0.3):
        """
        Instantiate a feed-forward neural network of the specified dimensions 
        and activation functions.Neuron layers do NOT include a bias term.

        layer_sizes:
            the number of neurons in each layer,
            including the input and output layers.

        activation_funcs:
            the activation functions to use in each layer.
            TODO: If a single function is provided, it is used for all non-input layers.
            If None, defaults to sigmoid activation.

        activation_func_gradients:
            functions for the gradient of the activation functions in each layer.
            The function is passed the output of a layer as argument.
            TODO: If a single function is provided, it is used for all non-input layers.
            Must be provided if a activation func is provided.
                TODO: could implement a numerical differences method.
            If None, defaults to sigmoid activation.

        loss_func:
            the callable loss function to use.
            It is passed the predicted and true label encodings.
            Note, this is not used for training.
            If None, defaults to square error loss.

        loss_func_gradient:
            a callable function for the gradient of the loss function,
            It is passed the predicted and true label encodings.
            Must be provided if loss_func is provided.
                TODO: could implement a numerical differences method.
            If None, defaults to the gradient of the square error loss.

        learning_rate:
            learning rate divisor. TODO: more on this.
        """
        assert len(layer_sizes) > 2, \
            ValueError(f"The number of layers ({len(layer_sizes)}) " +\
                       " must be >2 to contain an input and output layer.")

        self.layer_sizes = layer_sizes
        self.depth = len(layer_sizes) - 1  # Not counting the input layer
        self.learning_rate = learning_rate

        # Weights and biases
        # index 0 is the edges before layer 1.
        self.layer_weights = []
        self.biases = []
        for i_size, o_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layer_weights.append(np.random.uniform(-0.5, 0.5, (o_size, i_size)))
            self.biases.append(np.random.uniform(-0.5, 0.5, (o_size, 1)))

        
        # Assigns the activation functions of each layer and their derivatives
        #  The default function is the sigmoid function.
        if activation_funcs is None:
            self.activation_funcs = (sigmoid,) * self.depth
            self.activation_derivatives = (sigmoid_derivative, ) * self.depth

        else:
            assert self.depth == len(activation_funcs), \
                ValueError(f"The number of layers {len(layer_sizes)}-1={len(layer_sizes)-1} " + \
                            f"must be 1, or consistent with the number of activation functions ({len(activation_funcs)}).")
            self.activation_funcs = activation_funcs

            if activation_func_gradients is None:
                raise NotImplementedError("Not yet implemented")
                self.activation_derivatives = tuple(_numerical_gradient(fn) for fn in self.activation_funcs)
            else:
                assert self.depth == len(activation_func_gradients), \
                    ValueError(f"The number of layers {len(layer_sizes)}-1={len(layer_sizes)-1} " + \
                               f"must be consistent with the number of activation function derivatives ({len(activation_func_gradients)}).")
                self.activation_derivatives = activation_func_gradients


        # Assigns the loss functions and its derivative
        #  The default function is the square error loss function.
        if loss_func is None:
            self.loss_func = square_loss
            self.loss_func_derivative = square_loss_gradient  # This returns positive gradient
        else:
            self.loss_func = loss_func
            if loss_func_gradient is None:
                raise NotImplementedError("Not yet implemented")
                self.loss_func_derivative = _numerical_loss_gradient(self.loss_function)  # This returns positive gradient
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
        LB Note, is the issue not that the gradients are very small?
        """
        norm = np.linalg.norm(grad)
        return grad / (norm + eps)


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
            as 1 / (epoch_count + 1).
        """
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

        # Normalize and update the matrices!
        # following clip grad norm from pytorch (or can use global norm)
        # forcing each gradients to norm 1.0 prevents the step size from naturally shrinking as the network approaches a local minimum
        # causing updates to bounce around the minimum
        
        def clip_grad_norm(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray], max_norm: float = 1.0, eps: float = 1e-8):
            """
            Clips global gradient norm to max_norm. If global_norm <= max_norm,
            gradients remain completely unchanged.
            """
            total_sq_norm = sum(np.sum(w ** 2) for w in weight_grads) + \
                            sum(np.sum(b ** 2) for b in bias_grads)
            
            global_norm = np.sqrt(total_sq_norm)
            
            clip_coef = max_norm / (global_norm + eps)
            
            # Only scale down if global_norm exceeds max_norm
            if clip_coef < 1.0:
                weight_grads = [w * clip_coef for w in weight_grads]
                bias_grads = [b * clip_coef for b in bias_grads]
                
            return weight_grads, bias_grads
        
        # global norm option too
        def _normalize_global(self, weight_grads: list[np.ndarray], bias_grads: list[np.ndarray], eps: float = 1e-8):
            """
            Rescales all weight and bias gradients together using a single global norm,
            preserving relative gradient proportions across layers and parameters.
            """
            # Calculate sum of squared Frobenius norms across all matrices
            total_sq_norm = sum(np.sum(w ** 2) for w in weight_grads) + \
                            sum(np.sum(b ** 2) for b in bias_grads)
            
            global_norm = np.sqrt(total_sq_norm)
            
            scale = 1.0 / (global_norm + eps)
            
            weight_grads = [w * scale for w in weight_grads]
            bias_grads = [b * scale for b in bias_grads]
            
            return weight_grads, bias_grads


        weight_grads, bias_grads = clip_grad_norm(self, weight_grads, bias_grads)
        # Learning rate, decayed by epoch count
        # for larger training inverse decay
        # note that for small training fixed lr is sufficient
        decay_rate = 0.02 
        lr = self.learning_rate / (1.0 + decay_rate * epoch_count)

        # adjust weights
        for i in range(self.depth):
            self.layer_weights[i] -= lr * weight_grads[-i-1]
            self.biases[i] -= lr * bias_grads[-i-1]
