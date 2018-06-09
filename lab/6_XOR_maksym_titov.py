# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:21:50 2018

@author: Maxim
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

inputs = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]))
targets = list(map(lambda s: Variable(torch.Tensor([s])), [
    [0],
    [1],
    [1],
    [0]
]))

criterion = nn.MSELoss()

net = Network()
optimizer = optim.RMSprop(net.parameters(), lr=0.01)

print("Training with RMSprop...")
rms_idx = 0
rms_error = 1
while (True):
    rms_error= 0
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        rms_error += abs(target.data.numpy()[0] - output.data.numpy()[0])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    rms_idx+=1
    rms_error = 0.25*rms_error
    if rms_idx > 10000 or rms_error < 0.01:
        #print("Epoch {} Loss: {}, stop calculations".format(idx, loss.data.numpy()[0]))
        break
print("")
print("Final results for RMSprop:")
print("Epoch number: {}".format(rms_idx))
print("Avarage error: {}".format(rms_error))
for input, target in zip(inputs, targets):
    output = net(input)
    print("{} XOR {}: real={} predicted={} (error:{})".format(
        int(input.data.numpy()[0][0]),
        int(input.data.numpy()[0][1]),
        int(target.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
    ))


net = Network()
optimizer = optim.SGD(net.parameters(), lr=0.01)

print("Training with SGD...")
sgd_idx = 0
sgd_error = 1
while (True):
    sgd_error= 0
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()   # zero the gradient buffers
        output = net(input)
        sgd_error += abs(target.data.numpy()[0] - output.data.numpy()[0])
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()    # Does the update
    sgd_idx+=1
    sgd_error = 0.25*sgd_error
    if sgd_idx > 10000 or sgd_error < 0.01:
        #print("Epoch {} Loss: {}, stop calculations".format(idx, loss.data.numpy()[0]))
        break

print("")
print("Final results for SGD:")
print("Epoch number: {}".format(sgd_idx))
print("Avarage error: {}".format(sgd_error))
for input, target in zip(inputs, targets):
    output = net(input)
    print("{} XOR {}: real={} predicted={} (error:{})".format(
        int(input.data.numpy()[0][0]),
        int(input.data.numpy()[0][1]),
        int(target.data.numpy()[0]),
        round(float(output.data.numpy()[0]), 4),
        round(float(abs(target.data.numpy()[0] - output.data.numpy()[0])), 4)
    ))

sum_rms_idx = 0
sum_sgd_idx = 0
sum_rms_error = 0
sum_sgd_error = 0
STATISTIC_NUMBER = 10
print("Run statistic estimation...")
for statistic in range(0,STATISTIC_NUMBER):
    print("iteration number: {}".format(statistic))
    net = Network()
    optimizer = optim.RMSprop(net.parameters(), lr=0.01)
    
    rms_idx = 0
    rms_error = 1
    while (True):
        rms_error= 0
        for input, target in zip(inputs, targets):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(input)
            rms_error += abs(target.data.numpy()[0] - output.data.numpy()[0])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
        rms_idx+=1
        rms_error = 0.25*rms_error
        if rms_idx > 10000 or rms_error < 0.01:
            #print("Epoch {} Loss: {}, stop calculations".format(idx, loss.data.numpy()[0]))
            break
    sum_rms_idx += rms_idx
    sum_rms_error +=rms_error
    
    net = Network()
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    
    sgd_idx = 0
    sgd_error = 1
    while (True):
        sgd_error= 0
        for input, target in zip(inputs, targets):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(input)
            sgd_error += abs(target.data.numpy()[0] - output.data.numpy()[0])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update
        sgd_idx+=1
        sgd_error = 0.25*sgd_error
        if sgd_idx > 10000 or sgd_error < 0.01:
            #print("Epoch {} Loss: {}, stop calculations".format(idx, loss.data.numpy()[0]))
            break
    sum_sgd_idx += sgd_idx
    sum_sgd_error +=sgd_error
        
print("\nMain score")
print("--------SGD-----RMSProp---")
print("Epoch number: {}   {}".format(sum_sgd_idx/STATISTIC_NUMBER,sum_rms_idx/STATISTIC_NUMBER))
print("Avr. error: {}   {}".format(sum_sgd_error/STATISTIC_NUMBER,sum_rms_error/STATISTIC_NUMBER))
