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

def test_conv2d(ex, w, b, P, S):
    out_student = conv2d_forward(ex, w, b, P, S)
    out = pytorch_conv2d_foward(ex, w, b, P, S)
    result = np.allclose(out, out_student, atol=1e-2)
    return result

np.random.seed(777)
ex = x_train[0:40].view(40, 4, 14, 14)
w = torch.FloatTensor(np.random.uniform(size=(16, 4, 5, 5)))
b = torch.FloatTensor(np.random.uniform(size=(16,)))
results['1'] = test_conv2d(ex, w, b, 0, 1)


# TODO: Implement this function
import torch
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
    input_with_padding = input
    if padding>0:
        zero = torch.zeros([padding], dtype=torch.float32)
        input_with_padding = torch.zeros([input.shape[0],input.shape[1],W+padding*2,H+padding*2], dtype=torch.float32)
        for ind in range(0, N):
            new_sample =  torch.zeros([input.shape[1],W+padding*2,H+padding*2], dtype=torch.float32)
            for i in range(0,input.shape[1]):#(0..4)
                new_subsample = torch.zeros([W+padding*2,H+padding*2], dtype=torch.float32)
                for j in range(0,W):
                    new_subsample[j] = torch.cat((zero,input[ind][i][j], zero), 0)
                    new_sample[i][j+padding] = new_subsample[j]
            input_with_padding[ind] = new_sample
       
    
    w_size = int( (W-F+2*P)/S + 1 )
    h_size = int( (H-F+2*P)/S + 1 )
    out=torch.zeros(N, D, w_size, h_size)
    for sampl in range(N):
        for filt in range(D):
            for a in range(w_size):
                for b in range(h_size):
                    out[sampl, filt, a, b]=(input_with_padding[sampl,:,(a*S):(a*S+F),(b*S):(b*S+F)]*kernel[filt,:,:,:]).sum()+bias[filt]
    #out = np.random.uniform(size=(N, D, (W-F+2*P)/S+1, (H-F+2*P)/S+1))
    return out
def pytorch_conv2d_foward(input, kernel, bias, padding, stride):
    # Note: You don't have to change this function!
    # Ugly code to forward input through PyTorch convolution
    assert kernel.shape[-2] == kernel.shape[-1]
    kernel_size = kernel.shape[-1]
    n_filters = kernel.shape[0]
    n_channels = kernel.shape[1]
    m = nn.Conv2d(n_channels, n_filters, kernel_size=kernel_size, padding=padding, stride=stride)
    m.weight.data.copy_(kernel)
    m.bias.data.copy_(bias)
    output = m.forward(Variable(input))
    output = output.data.numpy()
    return output
np.random.seed(780)
ex = x_train[0:40].view(40, 1, 28, 28)
w = torch.FloatTensor(np.random.uniform(size=(16, 1, 2, 2)))
b = torch.FloatTensor(np.random.uniform(size=(16,)))
result = test_conv2d(ex, w, b, 4, 4)