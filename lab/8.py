import json
import matplotlib as mpl
from src import fmnist_utils
from src.fmnist_utils import *

import torch
from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
from itertools import repeat
from torch import nn
from torch import optim

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
for i in range(0, x_train.shape[1]):
    print(x_train[:,i])
    break

input_dim = 784
output_dim = 10
alpha = 0.93
hidden_dims =  [50, 50, 50]
model = torch.nn.Sequential()
previous_dim = input_dim
for id, D in enumerate(hidden_dims):
    model.add_module("dropout_{}".format(id), BernoulliDropout(alpha))
    model.add_module("linear_{}".format(id), torch.nn.Linear(previous_dim, D, bias=True))
    model.add_module("nonlinearity_{}".format(id), torch.nn.ReLU())
    previous_dim = D
model.add_module("final_layer", torch.nn.Linear(D, output_dim, bias=True))

loss = torch.nn.CrossEntropyLoss(size_average=True)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
H = train(loss=loss, model=model, x_train=x_train, y_train=y_train,
          x_test=x_test, y_test=y_test,
          optim=optimizer, batch_size=128, n_epochs=100)

plot(H)