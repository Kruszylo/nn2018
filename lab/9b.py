# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:42:01 2018

@author: Maxim
"""


import json
import matplotlib as mpl
from src import fmnist_utils
from src.fmnist_utils import *

from torchvision import utils
import numpy as np
import matplotlib.pyplot as plt

def plot(H):
    plt.title(max(H['test_acc']))
    plt.plot(H['acc'], label="acc")
    plt.plot(H['test_acc'], label="test_acc")
    plt.legend()

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['figure.figsize'] = (7, 7)
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['axes.labelsize'] = 12

(x_train, y_train), (x_test, y_test) = fmnist_utils.get_data(which="mnist")

x_train_4d = x_train.view(-1, 1, 28, 28)
x_test_4d = x_test.view(-1, 1, 28, 28)
import torch
class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def build_conv(input_dim, output_dim, n_filters=32, maxpool=4, hidden_dims=[32], dropout=0.0):
    model = torch.nn.Sequential()
    
    # Convolution part
    model.add_module("conv2d", torch.nn.Conv2d(input_dim[0], n_filters, kernel_size=5, padding=0))
    model.add_module("relu", torch.nn.ReLU()) 
    model.add_module("maxpool", torch.nn.MaxPool2d(maxpool))
    model.add_module("dropout", torch.nn.Dropout2d(dropout))
    model.add_module("flatten", Flatten()) # Add flattening from 4d -> 2d. 
    
    f = model(Variable(torch.ones(1,*input_dim)))
    previous_dim = int(np.prod(f.size()[1:]))
    #previous_dim = self.get_flat_fts(input_dim, self.model)
    
    # Classifier
    for id, D in enumerate(hidden_dims):
        model.add_module("linear_{}".format(id), torch.nn.Linear(previous_dim, D, bias=True))
        model.add_module("nonlinearity_{}".format(id), torch.nn.ReLU())
        previous_dim = D
    model.add_module("final_layer", torch.nn.Linear(D, output_dim, bias=True))
    return model

## Starting code for training a ConvNet.
input_dim = (1, 28, 28)

model = build_conv(input_dim, 10, n_filters=32, dropout=0.5)

loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

H = train(loss=loss, model=model, x_train=x_train_4d, y_train=y_train,
          x_test=x_test_4d, y_test=y_test,
          optim=optimizer, batch_size=128, n_epochs=50)

plot(H)
## Starting code for training a MLP.

#model = build_mlp(784, 10)
#loss = torch.nn.CrossEntropyLoss(size_average=True)
#optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
#H_mlp = train(loss=loss, model=model, x_train=x_train, y_train=y_train,
#          x_test=x_test, y_test=y_test, optim=optimizer, batch_size=128, n_epochs=100)
#plot(H_mlp)
## Starting code for Ex2.1

def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    '''
    vistensor: visuzlization tensor
        @ch: visualization channel 
        @allkernels: visualization all tensores
    ''' 
    
    n,c,w,h = tensor.shape
    if allkernels: tensor = tensor.view(n*c,-1,w,h )
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)
        
    rows = np.min( (tensor.shape[0]//nrow + 1, 64 )  )    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.detach().numpy().transpose((1, 2, 0)))
    plt.savefig("9b_1.png")

filters = torch.FloatTensor(model.conv2d.weight) # Get Torch Tensor corresponding to filters from the best validation point in training
vistensor(filters, ch=0, allkernels=True)