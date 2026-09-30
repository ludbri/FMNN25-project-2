"""
Generates adversarial examples against a trained NeuralNetwork by
gradient-ascent-style perturbation of an input image until the network
classifies it as a chosen target class.
"""

import numpy as np
from NeuralNetwork import NeuralNetwork
import parameters
import matplotlib.pyplot as plt
from binary import digit_to_binary
from parameters import BINARY_ENCODING


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

    if BINARY_ENCODING:
        bits = np.asarray(digit_to_binary(target), dtype='d').reshape(-1,1) #(4,1)
        grad_target = 2.0 * bits - 1.0 #Bits that should conver to 1 become +1, bits that should become 0 are set to -1
    else:
        # One-hot vector representing the desired (target) output 
        grad_target = np.zeros((parameters.N_CLASSES,1), dtype='d')
        grad_target[target] = 1
    # grad_target[y] = -1   # TODO: try to remove this line! What happens? <- it does not seem to make a visual difference.
    # When I try an attack with 4 bit encoding WITHOUT normalizing the gradient
    # I get a much better (visually more accurate) result
    x = np.asarray(x).reshape(-1, network.layer_sizes[0])

    max_iters = 100000
    i = 0
    while target != network.predict(x) and i<max_iters:
        i += 1
        post_activations, pre_activations = network.forward(x.T)

        # backpropogate gradient of o w.r.t. x
        grad = grad_target
        for w, act, actgrad in zip(network.layer_weights[::-1],
                                             pre_activations[::-1],
                                             network.activation_derivatives[::-1]
                                             ):
            # gradient of output w.r.t. the input of this layer
            a_grad = grad * actgrad(act)
            # gradient w.r.t. the output of the previous layer
            grad = w.T @ a_grad

        # Normalize the size to 1
        #grad /= np.linalg.norm(grad, 2)

        x += stsize * grad.T
        x = np.clip(x, 0., 1.)

        if i % 100 == 0 and BINARY_ENCODING:
            print(f"target: {target}, i: {i} xsum {x.sum()}")            
        elif i % 1000 == 0: 
            print(f"target: {target}, i: {i} xsum {x.sum()}")
    return x



def make_attacks(network: NeuralNetwork,
                 x: np.ndarray,
                 fig=None,
                 title=None,
                 att: str = "target_attack"):
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
    attack: str, default="target_attack"
        The attack chosen to be used
        "target_attack" : specifies a targeted attack
        "negative_gradient" : specifies taking a step in the negative gradient direction

    Returns
    -------
    None
        Displays a matplotlib figure; does not return a value.
    """
    xs = []
    for y_target in range(parameters.N_CLASSES):
        if att == "target_attack":
            xs.append(attack(network, x, y_target))
            
        elif att == "negative_gradient":
            xs.append(attack_negative_gradient(network, x))
            
        else:
            raise ValueError(f"Unknown attack: {att}")

    input_fig_is_None = fig is None
    if input_fig_is_None:
        fig = plt.figure(figsize=(16,10))

    axes = fig.subplots(3,5)
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

    # fig.tight_layout()
    if input_fig_is_None:
        plt.show()
    
    
def attack_negative_gradient(
        network: NeuralNetwork, 
        x: np.ndarray,
        step_size: float = 0.1,
        max_iters: int = 200
    ) -> np.ndarray:
    '''
    Performs an untargeted attack by taking steps along the negative gradient
    of the currently predicted class, pushing the network to change its prediction.

    Parameters
    ----------
    network : NeuralNetwork
        The trained network to attack (its weights are not modified;
        only the input `x` is perturbed).
    x : np.ndarray
        Starting input image of shape (1, input_size) or flat array.
    step_size : float, default=0.01
        Magnitude of perturbation added per iteration.
    max_iters : int, default=200
        Maximum iterations before stopping
    Returns
    -------
    np.ndarray
        The perturbed input imagePerturbed input image clipped to valid range [0, 1].
    '''
    
    x = np.asarray(x).copy().reshape(1, network.layer_sizes[0])
    initial_pred = network.predict(x)[0]
    
    for _ in range(max_iters):
        current_pred = network.predict(x)[0]
        
        # stop once the netwrok prediction changes from initial (if before max iterations)
        if current_pred != initial_pred:
            break
        
        
        # One-hot vector representing the current predicted class
        y_onehot = np.zeros((parameters.N_CLASSES, 1))
        y_onehot[current_pred] = 1.0 
        
        # forward pass to obtain activations
        activations, pre_activations = network.forward(x.T)
        
        # compute gradient of the loss w.r.t. predicted class
        backgrad = network.loss_func_derivative(activations[-1], y_onehot)
        
        # backpropagate gradient w.r.t input layer
        for w, z, actgrad in zip(
                network.layer_weights[::-1],
                pre_activations[::-1],
                network.activation_derivatives[::-1]):
            a_grad = backgrad * actgrad(z)
            backgrad = w.T @ a_grad
        
        grad_x = backgrad.T
        
        x += step_size * np.sign(grad_x)
        x = np.clip(x,0.0, 1.0)
        
        return x
