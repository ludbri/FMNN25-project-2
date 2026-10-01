# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 18:21:06 2026
@author: annajobbar
"""
import numpy as np
from typing import Callable

# An activation function evaluates a matrix of inputs element-by-element.
ActivationFunc = Callable[[np.ndarray], np.ndarray]
# An activation function gradient evaluates the gradient for a matrix of layer outputs, element-by-element.
ActivationFuncGrad = Callable[[np.ndarray], np.ndarray]
# A loss function takes matrices of the predicted and true label encodings and returns a float
LossFunc = Callable[[np.ndarray,np.ndarray], float]
# A loss function gradient takes matrices of predicted and true label encodings
#   and returns a vector of the gradient with respect to the network output
LossFuncGrad = Callable[[np.ndarray,np.ndarray], np.ndarray]


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
        The (positive) gradient of the mean squared error for each sample. 
    """
    return y_pred - y_true


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



def zero_one_loss(y_pred: np.ndarray,
                y_true: np.ndarray) -> float:
    """
    Calculates the zero-one loss function between predicted and true encoded values.
    Each sample has a loss of 0 if it is correctly predicted and 0 otherwise.

    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values, of shape (output_size,n) for n samples.
    y_true : np.ndarray
        Ground-truth values, of the same shape as `y_pred`.

    Returns
    -------
    float
        The zero-one loss.
    """
    y_pred_class = np.argmax(y_pred, axis=0)
    y_true_class = np.argmax(y_true, axis=0)
    n_wrong = np.array(y_pred_class != y_true_class, dtype=int)
    loss = n_wrong.sum()
    return loss

def zero_one_surrogate_gradient(y_pred: np.ndarray,
                            y_true: np.ndarray) -> np.ndarray:
    """
    Calculates a leaky gradient of the zero-one loss between predicted and true values.
    Uses a scaling of the absolute error of the one-hot encodings as surrogate function.

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
    eps = 1
    loss = eps * (y_pred - y_true)
    return loss
