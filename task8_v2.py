"""
Task 8: how closely can a one-hidden-layer network fit the first 50 MNIST
training samples?

Uses the quadratic loss on the RAW network output (no rounding/argmax),
10 output neurons, and stops each run at the first epoch where the training
loss falls below THRESHOLD.

Competition ranking: smallest hidden layer that reaches the threshold, then
fewest epochs, then smallest final loss.
"""

import numpy as np
from functools import partial

import parameters
from NeuralNetwork import NeuralNetwork, sigmoid, sigmoid_derivative, relu, relu_derivative
from dataloading import load_mnist, minibatches

# Task 8 requires 10 output neurons, so binary encoding must be OFF.
# (Set BINARY_ENCODING = False in parameters.py and restart the kernel.)
assert not parameters.BINARY_ENCODING and parameters.OUTPUT_SIZE == 10, \
    "Set BINARY_ENCODING = False in parameters.py (and restart the console)."

THRESHOLD = 1e-8
N_SAMPLES = 50

def identity(x):
    return x

def identity_derivative(output):
    return np.ones_like(output)


def step(x):
    return np.where(x>0, 1, 0)

def leaky_step_derivative(x,grad):
    return grad*np.ones_like(x)

def mod_step(x, grad):
    """
    returns
        grad*x + 1  for x<-offset,
        0           for -offset<x<0, and
        1           for x>0,
    where offset = 1/grad
    """
    return np.where(x>0, 1, np.where(x<-(1/grad),grad*x+1,0))

def mod_step_surrogate_derivative(x, grad):
    """derivative of the surrogate function f(x) = grad*x + 1."""
    return grad * np.ones_like(x)


def train_until_threshold(train_data, hidden_size, learning_rate,
                          mini_batch_size, max_epochs, seed=0):
    """
    Trains a fresh network and returns (epoch_reached, final_loss, network).
    epoch_reached is the first epoch (1-based) with loss < THRESHOLD, or None.
    """
    np.random.seed(seed)
    grad = 1
    network = NeuralNetwork(
                            layer_sizes=(parameters.INPUT_SIZE, 
                                         hidden_size, 
                                         parameters.OUTPUT_SIZE),
                            
                            # sigmoid hidden layer, linear output layer
                            activation_funcs=(sigmoid, partial(mod_step, grad=grad)),
                            activation_func_gradients=(sigmoid_derivative,
                                                       partial(mod_step_surrogate_derivative, grad=grad)),
                            learning_rate=learning_rate,
                            )

    # All 50 samples as one batch, used only to evaluate the loss
    x_all, y_all = next(minibatches(train_data, n=N_SAMPLES, one_hot=True))

    loss = network.evaluate_loss(x_all, y_all)
    for epoch in range(1, max_epochs + 1):
        for x, y in minibatches(train_data, batch_size=mini_batch_size,
                                n=N_SAMPLES, one_hot=True, shuffle=True):
            network.learn_batch(x, y, epoch_count=epoch - 1)

        loss = network.evaluate_loss(x_all, y_all)
        if not np.isfinite(loss):        # diverged, give up on this setting
            return None, loss, network
        if loss < THRESHOLD:
            return epoch, loss, network

    return None, loss, network


if __name__ == "__main__":
    print("Loading MNIST dataset...")
    training_data, _, _ = load_mnist("mnist.pkl")
    train_50 = (training_data[0][:N_SAMPLES], training_data[1][:N_SAMPLES])

    # ------------------------------------------------------------------
    # STEP 1: find ANY setting that reaches the threshold (comfortable size)
    # ------------------------------------------------------------------
    epoch, loss, _ = train_until_threshold(
        train_50, hidden_size=30, learning_rate=0.2,
        mini_batch_size=5, max_epochs=10000)
    print(f"Single test run: epoch reached = {epoch}, final loss = {loss:.3e}")

    # ------------------------------------------------------------------
    # STEP 2: shrink the hidden layer, trying a few learning rates each
    # ------------------------------------------------------------------
    hidden_sizes = range(10, 0, -1)
    learning_rates = (0.1, 0.2, 1, 10, 10**2, 10**3)
    mini_batch_sizes = (1,2, 5, 10, 25, 50)
    max_epochs = 5000

    winners = []   # (hidden_size, epochs, final_loss, lr, batch)
    for hidden_size in hidden_sizes:
        found = []
        best_epochs = max_epochs
        for lr in learning_rates:
            for mb in mini_batch_sizes:
                epoch, loss, _ = train_until_threshold(
                    train_50, hidden_size, lr, mb, best_epochs)
                if epoch is not None:
                    found.append((hidden_size, epoch, loss, lr, mb))
                    best_epochs = epoch
        if not found:
            print(f"hidden={hidden_size}: no setting reached {THRESHOLD:g}")
            break                      # smaller layers are unlikely to work
        best = min(found, key=lambda r: (r[1], r[2]))
        winners.append(best)
        print(f"hidden={best[0]}: epochs={best[1]}, loss={best[2]:.3e}, "
              f"lr={best[3]}, batch={best[4]}")

    if winners:
        # smallest hidden size, then fewest epochs, then smallest loss
        best = min(winners, key=lambda r: (r[0], r[1], r[2]))
        print("\nBest found (hidden, epochs, loss, lr, batch):", best)