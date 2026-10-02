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

    if network.binary_encoding:
        bits = np.asarray(digit_to_binary(target), dtype='d').reshape(-1,1) #(4,1)
        grad_target = 2.0 * bits - 1.0 #Bits that should conver to 1 become +1, bits that should become 0 are set to -1
    else:
        # One-hot vector representing the desired (target) output 
        grad_target = np.zeros((parameters.N_CLASSES,1), dtype='d')
        grad_target[target] = 1

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

        if i % 1000 == 0: 
            print(f"target: {target}, i: {i} xsum {x.sum()}")
    return x



def make_attacks(network: NeuralNetwork,
                 x: np.ndarray,
                 att: str = "target_attack"):
    """
    Runs the specified attack against the network for a single starting image,
    then displays the original image alongside the adversarial result(s).
    """
    axsize = int(np.sqrt(x.size))
    x_input = x.reshape(1, -1) if x.ndim == 1 else x

    if att == "target_attack":
        # 1. Collect all targeted attacks FIRST
        xs = []
        for y_target in range(parameters.N_CLASSES):
            xs.append(attack(network, x_input, y_target))

        # 2. Plot ONCE after all 10 examples are generated
        fig, axes = plt.subplots(3, 5, figsize=(16, 10))

        # Top row: original image centered
        axes[0, 2].imshow(x.reshape(axsize, axsize), cmap="Greys")
        axes[0, 2].set_title(f"start, predicted: {network.predict(x_input)[0]}")
        for i in (0, 1, 3, 4):
            axes[0, i].set_visible(False)

        # Middle row: first 5 target classes (0-4)
        for i, xi in enumerate(xs[:5]):
            xi_input = xi.reshape(1, -1)
            axes[1, i].imshow(xi.reshape(axsize, axsize), cmap="Greys")
            axes[1, i].set_title(f"predicted: {network.predict(xi_input)[0]}")

        # Bottom row: next 5 target classes (5-9)
        for i, xi in enumerate(xs[5:]):
            xi_input = xi.reshape(1, -1)
            axes[2, i].imshow(xi.reshape(axsize, axsize), cmap="Greys")
            axes[2, i].set_title(f"predicted: {network.predict(xi_input)[0]}")

        fig.tight_layout()
        plt.show()

    elif att == "negative_gradient":
        # Untargeted attack runs only ONCE (no y_target loop needed)
        x_adv = attack_negative_gradient(network, x_input, step_size=0.01)
        x_adv_input = x_adv.reshape(1, -1)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(x.reshape(axsize, axsize), cmap="Greys")
        axes[0].set_title(f"Original: {network.predict(x_input)[0]}")

        axes[1].imshow(x_adv.reshape(axsize, axsize), cmap="Greys")
        axes[1].set_title(f"Adversarial: {network.predict(x_adv_input)[0]}")

        fig.tight_layout()
        plt.show()

    else:
        raise ValueError(f"Unknown attack: {att}")
   
        

    
    
    
def attack_negative_gradient(
        network: NeuralNetwork, 
        x: np.ndarray,
        step_size: float = 0.01,
        max_iters: int = 10000
    ) -> np.ndarray:
    '''
    Performs an untargeted attack by taking steps along the negative gradient
    of the currently predicted class, pushing the network to change its prediction
    
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
        if network.binary_encoding:
            y_cur = np.asarray(digit_to_binary(int(current_pred)), dtype=float).reshape(-1, 1)
        else:
            y_cur = np.zeros((parameters.N_CLASSES, 1))
            y_cur[current_pred] = 1.0
        
        # forward pass to obtain activations
        activations, pre_activations = network.forward(x.T)
        
        # compute gradient of the loss w.r.t. predicted class
        backgrad = network.loss_func_derivative(activations[-1], y_cur)
        
        # backpropagate gradient w.r.t input layer
        for w, z, actgrad in zip(
                network.layer_weights[::-1],
                pre_activations[::-1],
                network.activation_derivatives[::-1]):
            a_grad = backgrad * actgrad(z)
            backgrad = w.T @ a_grad
        
        grad_x = backgrad.T
        
        x += step_size * np.sign(grad_x)  # * grad_x
        x = np.clip(x,0.0, 1.0)
        
    return x
    
    
    
    
