"""
Generates adversarial examples against a trained NeuralNetwork by
gradient-ascent-style perturbation of an input image until the network
classifies it as a chosen target class.
"""

import numpy as np
from NeuralNetwork import NeuralNetwork
import parameters
import matplotlib.pyplot as plt


def attack(network: NeuralNetwork,
           x: np.ndarray,
           target: int):
     """
    Minimally adjusts `x` via iterative gradient steps until the network
    classifies it as `target`.

    Parameters
    ----------
    network : NeuralNetwork
        The trained network to attack (its weights are not modified;
        only the input `x` is perturbed).
    x : np.ndarray
        Starting input image, flattened to shape matching
        `network.input_size`.
    target : int
        The target class index the attack tries to make the network
        predict for the (perturbed) input.

    Returns
    -------
    np.ndarray
        The perturbed input, reshaped to (1, network.input_size), for
        which `network.predict` returns `target`. Values are clipped
        to the valid image range [0, 1].
    """
    stsize = 10**-1
    x = x.copy()
    y = network.predict(x)
    
    # One-hot vector representing the desired (target) output 
    grad_target = np.zeros((parameters.N_CLASSES,1), dtype='d')
    grad_target[target] = 1
    # grad_target[y] = -1   # TODO: try to remove this line! What happens? <- it does not seem to make a visual difference.
    x = np.asarray(x).reshape(-1, network.input_size)

    o_prev = np.zeros_like(grad_target)
    i = 0
    while target != (y := network.predict(x)):
        i += 1
        h, o = network.forward(x.T)
        o_diff = o-o_prev

        # grad of o wrt x
        grad = grad_target
        # grad =  # o - y
        grad = grad * network.activation_derivative(o)
        grad = network.weights_hidden_output.T @ grad
        grad = grad * network.activation_derivative(h)
        grad = network.weights_input_hidden.T @ grad

        # Normalize the size to 1
        grad /= np.linalg.norm(grad, 2)

        x += stsize * grad.T
        x = np.clip(x, 0., 1.)

        if i % 1000 == 0: 
            print(f"target: {target}, i: {i} xsum {x.sum()}")
            o_prev = o.copy()
    return x



def make_attacks(network: NeuralNetwork,
                 x: np.ndarray):
    """
    Runs the `attack` function against every possible target class for
    a single starting image, then displays the original image alongside
    all resulting adversarial examples in a grid, each labeled with the
    network's predicted class.

    Parameters
    ----------
    network : NeuralNetwork
        The trained network to attack.
    x : np.ndarray
        A single starting input image (flattened).

    Returns
    -------
    None
        Displays a matplotlib figure; does not return a value.
    """
    xs = []
    for y_target in range(parameters.N_CLASSES):
        xs.append(attack(network, x, y_target))

    fig, axes = plt.subplots(3,5, figsize=(16,10))
    axsize = int(np.sqrt(x.size))
    axes[0,2].imshow(x.reshape(axsize, axsize), cmap="Greys")
    axes[0,2].set_title(f"start, predicted: {network.predict(x)}")
    for i in (0,1,3,4):
        axes[0,i].set_visible(False)
    for i, xi in enumerate(xs[:5]):
        axes[1,i].imshow(xi.reshape(axsize, axsize), cmap="Greys")
        axes[1,i].set_title(f"start, predicted: {network.predict(xi)}")
    for i, xi in enumerate(xs[5:]):
        axes[2,i].imshow(xi.reshape(axsize, axsize), cmap="Greys")
        axes[2,i].set_title(f"start, predicted: {network.predict(xi)}")

    fig.tight_layout()

    plt.show()
