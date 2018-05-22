# -*- coding: utf-8 -*-
"""
Created on Mon May 21 08:44:02 2018

@author: Maxim
"""

import json
import matplotlib as mpl
from src import fmnist_utils
from src.fmnist_utils import *

def plot(H):
    plt.title(max(H['test_acc']))
    plt.plot(H['acc'], label="acc")
    plt.plot(H['test_acc'], label="test_acc")
    plt.legend()

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (7, 7)
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12

(x_train, y_train), (x_test, y_test) = fmnist_utils.get_data()

np.random.seed(777)
ex = x_train[0:40].view(40, 4, 14, 14)
w = torch.FloatTensor(np.random.uniform(size=(16, 4, 5, 5)))
b = torch.FloatTensor(np.random.uniform(size=(16,)))
results['1'] = test_conv2d(ex, w, b, 0, 1)

def test_conv2d(ex, w, b, P, S):
    out_student = conv2d_forward(ex, w, b, P, S)
    out = pytorch_conv2d_foward(ex, w, b, P, S)
    result = np.allclose(out, out_student, atol=1e-2)
    return result

# TODO: Implement this function
def conv2d_forward(input, kernel, bias, padding, stride):
    """
    Params
    ------
    input: torch.FloatTensor, shape (n_examples, n_channels, width, height)
    kernel: torch.FloatTensor, shape (n_filters, n_channels, kernel_size, kernel_size)
    bias: torch.FloatTensor, shape (n_filters)
    padding: int
        Padding to add
    """
    # Dummy implementation sampling output with a correct shape
    N = input.shape[0] #батч 
    D = kernel.shape[0]
    W, H = input.shape[2], input.shape[3]
    F = kernel.shape[-1]
    S = stride #шфг вниз и вправо
    P = padding #ширина паддинга, заполняем нулями
    # TODO: Implement out!
    w_tensor = torch.zeros([W], dtype=torch.float32)
    h_tensor = torch.zeros([H+2], dtype=torch.float32)
    for ind in range(0, N):
        sample = input[ind]
        
    out = np.random.uniform(size=(N, D, (W-F+2*P)/S+1, (H-F+2*P)/S+1))
    return out