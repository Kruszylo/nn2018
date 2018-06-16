# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:31:14 2018

@author: Maxim
"""

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pylab as plt
from collections import defaultdict
import json
#torch.manual_seed(1)
def orthogonal(tensor, gain=1):
    # Code adapted from https://github.com/alykhantejani/nninit/blob/master/nninit.py
    if isinstance(tensor, Variable):
        orthogonal(tensor.data, gain=gain)
        return tensor
    else:
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported.")

        flattened_shape = (tensor.size(0), int(np.prod(tensor.numpy().shape[1:])))
        flattened = torch.Tensor(flattened_shape[0], flattened_shape[1]).normal_(0, 1)

        u, s, v = np.linalg.svd(flattened.numpy(), full_matrices=False)
        if u.shape == flattened.numpy().shape:
            tensor.view_as(flattened).copy_(torch.from_numpy(u))
        else:
            tensor.view_as(flattened).copy_(torch.from_numpy(v))

        tensor.mul_(gain)
        return tensor
    
# Hyper Parameters (constant for the notebook)
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 64
LR = 0.001
TIME_STEP = 28          # rnn time step / image height
INPUT_SIZE = 28         # rnn input size / image width
#                                           LR = 0.01               # learning rate
DOWNLOAD_MNIST = True   # set to True if haven't download the data

# A standard way to load a dataset
train_data = dsets.MNIST(
    root='./mnist/',
    train=True,                         # this is training data
    transform=transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,            # download it if you don't have it
)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# convert test data into Variable, pick 2000 samples to speed up testing
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# shape (2000, 28, 28) value in range(0,1)
test_x = Variable(test_data.test_data, volatile=True).type(torch.FloatTensor)[:2000]/255.   
test_y = test_data.test_labels.numpy().squeeze()[:2000]    # covert to numpy array

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_strategy="simple", init_scale=0.01):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear((input_size + hidden_size), hidden_size) ## Hint: linear module, from (input size + hidden_size) to hidden_size
        self.h2o = nn.Linear(hidden_size, output_size) # Hint: linear module
        
        if init_strategy == "simple":
            self.i2h.weight.data.normal_(0, init_scale)
        elif init_strategy == "orth":
            orthogonal(self.i2h.weight)
        else:
            raise NotImplementedError()
        
        self.i2h.bias.data.fill_(0)
        self.h2o.weight.data.uniform_(0, init_scale)
        self.h2o.bias.data.fill_(0)
            

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1) # Hint: use torch.cat to combine input and hidden input 2d vector
        hidden = F.tanh(self.i2h(combined)) # Hint: use input to hidden
        output = F.softmax(self.h2o(hidden)) # Hint: use hidden to output
        return output, hidden
    

    def initHidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size))
rnn = RNN(28, 64, 10)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()
# training and testing
H = {"acc": []}
EPOCH = 0
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y
                
        hidden = rnn.initHidden(b_x.size()[0])
        for i in range(b_x.size()[1]): # Hint: iterate through all the steps
            output, hidden = rnn.forward(b_x[:,i,:], hidden) # Hint: just apply forward from the model
        loss = loss_func(output, b_y)
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward()                                 # backpropagation, compute gradients
        optimizer.step()                                # apply gradients

        if step % 50 == 0:
            hidden = rnn.initHidden(test_x.size()[0])
            for i in range(test_x.size()[1]):
                test_output, hidden = rnn(test_x[:, i], hidden)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y.reshape(-1,)) / float(test_y.size)
            H['acc'].append(accuracy)
            print('Epoch: ', epoch + step*len(b_x)/2000., '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)
            
plt.title("Simple(0.01)")
plt.plot(H['acc'])
plt.xlabel("Epoch")
plt.ylabel("Training accuracy")
plt.savefig("10a_1.png")
thidden = rnn.initHidden(test_x.size()[0])
for i in range(test_x.size()[1]):
    ttest_output, thidden = rnn(test_x[:, i], thidden)
pred_y = torch.max(ttest_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10], 'real number')
###############################################################################
H = defaultdict(list)

rnn = RNN(28, 64, 10, "simple", 0.001)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted
EPOCH = 1
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):        # gives batch data
        b_x = Variable(x.view(-1, 28, 28))              # reshape x to (batch, time_step, input_size)
        b_y = Variable(y)                               # batch y
        
        optimizer.zero_grad()
        
        # Make sure gradients are retained
        hidden = rnn.initHidden(b_x.size()[0])
        hiddens = []
        for i in range(b_x.size()[1]): # Hint: iterate through all the steps
            output, hidden = rnn.forward(b_x[:,i,:], hidden) # Hint: just apply forward from the model
            hiddens.append(hidden)
            hidden.retain_grad() # google "retain_grad". Otherwise hidden.grad=None. 
        loss = loss_func(output, b_y)
        
        optimizer.zero_grad()                           # clear gradients for this training step
        loss.backward(retain_graph=True)      # backpropagation, compute gradients
        
        # Save ||dL/dh_i||
        dLdhi = []
        for h in hiddens[1:]:
            g = h.grad.data.numpy()
            dLdhi.append(np.linalg.norm(g)) # Hint: Just compute ||dL/dh_i|| from h.grad.data.numpy() using np.linalg.norm + average over examples
        
        optimizer.step()                                

        if step % 50 == 0:
            hidden = rnn.initHidden(test_x.size()[0])
            for i in range(test_x.size()[1]):
                test_output, hidden = rnn(test_x[:, i], hidden)
            
            for i, val in enumerate(dLdhi):
                H['dL/dh_{}'.format(i)].append(val)
                
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y.reshape(-1,)) / float(test_y.size)
            H['acc'].append(accuracy)
            print('Epoch: ', epoch + step*len(b_x)/2000., '| train loss: %.4f' % loss.data[0], '| test accuracy: %.2f' % accuracy)

# Save results
cm = plt.get_cmap("coolwarm", 80)
plt.title("Simple(0.01)") 
for i in range(28):
    plt.plot(H['dL/dh_{}'.format(i)], color=cm(i/28.))
plt.savefig("10a_2_simple.png") # Change to 2_orth.png for orthonormal run