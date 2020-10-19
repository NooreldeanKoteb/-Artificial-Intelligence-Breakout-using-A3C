# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 17:04:57 2020

@author: Nooreldean Koteb

"""

#AI for Breakout

#Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#Initializing and setting the variance of the tensor weights
#Initializes weights with a specific standard deviation
def normalized_columns_initializer(weights, std = 1.0):
    #Output torch tensor with random distribution of weights
    out = torch.randn(weights.size())
    
    #Normalization of weights using standard deviation
    #Variance(out) = std^2
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True).expand_as(out))
    
    #Return normalized output
    return out

#Initializing the weights of the neural network for an optimal learning
#m = neural network
def  weights_init(m):
    #Getting the classname of the neural network
    classname = m.__class__.__name__

    #If the connection is a convolution
    if classname.find('Conv') != -1:
        #then we will do a special initialization of the weights
        weight_shape = list(m.weight.data.size())
        
        #Product of the dimensions of the weights dim1*dim2*dim3
        fan_in = np.prod(weight_shape[1:4])
        #Product of the dimensions of the weights dim0*dim2*dim3
        fan_out = np.prod(weight_shape[2:4])*weight_shape[0]
        
        #Size of tensor of weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))
    
        #Inversely proportional weights using the two bounds
        m.weight.data.uniform_(-w_bound, w_bound)
        
        #Initializing the bias with 0s
        m.bias.data.fill_(0) 
    
    #If the connection is linear
    elif classname.find('Linear') != -1:
        #then we will do a special initialization of the weights
        weight_shape = list(m.weight.data.size())
        
        #dimension of the weights dim1
        fan_in = weight_shape[1]
        #dimension of the weights dim0
        fan_out = weight_shape[0]
    
        #Size of tensor of weights
        w_bound = np.sqrt(6. / (fan_in + fan_out))
    
        #Inversely proportional weights using the two bounds
        m.weight.data.uniform_(-w_bound, w_bound)
        
        #Initializing the bias with 0s
        m.bias.data.fill_(0) 
    


#Making the A3C brain
class ActorCritic(torch.nn.Module):
    
    #Initializing the class with number of inputs, and possible actions
    def __init__(self, num_inputs, action_space):
        #Inheriting Torch.nn.Module
        super(ActorCritic, self).__init__()
        
        #Convolutions
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride = 2, padding = 1)
        
        #RNN - LSTM
        #(input from conv, number of output neurons)
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)
        
        #Number of possible outputs
        num_outputs = action_space.n
        
        #Linear full connection for critic
        #(input neurons, output neurons)
        self.critic_linear = nn.Linear(256, 1) #Output = V(S)

        #Linear full connection for actor
        #(input neurons, actions outputs)
        self.actor_linear = nn.Linear(256, num_outputs) #Output = Q(S,A)
        
        #initialize weights and bias
        self.apply(weights_init)
        
        #Small std variation for the actor and large std variation for the critic
        #Changing this balances exploration vs exploitation
        
        #Initializing actor Weights
        self.actor_linear.weight.data = normalized_columns_initializer(self.actor_linear.weight.data,
                                                                       std = 0.01)
    
        #Initializing actor bias (kinda useless since its already done in the function above)
        self.actor_linear.bias.data.fill_(0)
        
        #Initializing critic Weights
        self.critic_linear.weight.data = normalized_columns_initializer(self.critic_linear.weight.data,
                                                                       std = 1.0)
    
        #Initializing critic bias (kinda useless since its already done in the function above)
        self.critic_linear.bias.data.fill_(0)
    
        #Initializing bias of lstm
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        
        #Puts model in train mode
        self.train()
    
    #Forward propagation
    #inputs = inputs, hidden nodes, and cell nodes
    def forward(self, inputs):
        #inputs, (hidden states, cell states)
        inputs, (hx, cx) = inputs
        
        #First layer | elu non-linear activation function (exponential linear unit)
        x = F.elu(self.conv1(inputs))
        #Second layer
        x = F.elu(self.conv2(x))
        #Third layer
        x = F.elu(self.conv3(x))
        #Fourth layer
        x = F.elu(self.conv4(x))
        
        #Flattening step
        #One dimensional vector
        x = x.view(-1, 32 * 3 * 3)
        
        #Putting values through lstm
        hx, cx = self.lstm(x, (hx, cx))
        
        #Updating x with output of hidden nodes
        x = hx
        
        #Returning the linear full connections of critic and actor & (hidden nodes, cell nodes)
        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
        
        
        
        
        
    
    
    
