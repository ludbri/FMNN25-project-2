import numpy as np
from NeuralNetwork import NeuralNetwork
import parameters


def attack(network: NeuralNetwork,
           x: np.array,
           target: int):
    """
    Minimally adjust x such that the network believes it is of class target.
    """
    stsize = 10**3
    x = x.copy()
    grad_target = -1*np.zeros((parameters.N_CLASSES,1), dtype='d')
    grad_target[target] = 1
    x = np.asarray(x).reshape(-1, network.input_size)
    o_prev = np.zeros_like(grad_target)
    while target != (y := network.predict(x)):
       h, o = network.forward(x.T)
       o_diff = o-o_prev

       # grad of o wrt x
       grad = grad_target
       # grad =  # o - y
       grad = grad * network.activation_derivative(o)
       grad = network.weights_hidden_output.T @ grad
       grad = grad * network.activation_derivative(h)
       grad = network.weights_input_hidden.T @ grad

       x += stsize * grad.T
       x = np.clip(x,0.,1.)

       print(f"xsum {x.sum()}")
       o_prev = o.copy()