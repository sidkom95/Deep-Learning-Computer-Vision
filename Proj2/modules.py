# -*- coding: utf-8 -*-
"""Modules.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BWVjsr3aUcRHdq70c1X3zoBJq5n6K3n9
"""

import torch
import math
from torch import Tensor, FloatTensor

#Base Class Module

class Module(object):
  '''
  Base class for other neural network modules to inherit from.
  '''

  def __init__(self):
    self._author = 'Ahmed Ben Haj Yahia, Amir Ghali, Mahmoud Sellami.'
  
  def forward(self, *input):
    '''
    Should get for input, and returns, a tensor or a tuple of tensors.
    '''
    raise NotImplementedError

  def backward(self, *gradwrtoutput):
    '''
    Should get as input a tensor or a tuple of tensors containing the gradient of the loss
    with respect to the module’s output, accumulate the gradient wrt the parameters, and return a
    tensor or a tuple of tensors containing the gradient of the loss wrt the module’s input.
    '''
    raise NotImplementedError

  def param(self):
    '''
    Should return a list of pairs, each composed of a parameter tensor, and a gradient tensor
    of same size. This list should be empty for parameterless modules (e.g. ReLU).
    '''
    return []

#Module Class: Linear

class Linear(Module):
  '''
  Fully Connected Layer defined by its input and output dimensions
  '''

  def __init__(self, input_dim, output_dim, mean_value = 0, std_value = 1):
    super().__init__()
    self.x = 0
    self.w = Tensor(output_dim, input_dim).normal_(mean = mean_value, std = std_value)
    self.b = Tensor(output_dim).normal_(mean = mean_value, std = std_value)
    self.dl_w = Tensor(self.w.size())
    self.dl_b = Tensor(self.b.size())
  
  def forward(self, input):
    self.x = input
    return self.w.mv(self.x) + self.b

  def backward(self, gradwrtoutput):
    self.dl_w.add_(gradwrtoutput.view(-1,1).mm(self.x.view(1,-1)))
    self.dl_b.add_(gradwrtoutput)
    return self.w.t().mv(gradwrtoutput)

  def param(self):
    return [(self.w, self.dl_w),(self.b, self.dl_b)]

#Module Class: ReLU

class ReLU(Module):
  '''
  Activation Function: x → max(0, x)
  '''

  def __init__(self):
    super().__init__()
    self.x = 0
    
  def forward(self, input):
    self.x = input
    return self.x.clamp(min=0)
 
  def backward(self, gradwrtoutput):
    dl_relu = self.x.sign().clamp(min=0)
    return dl_relu * gradwrtoutput

  def param(self):
    return [(None,None)]

#Module Class: Tanh

class Tanh(Module):
  '''
  Activation Function: x → [2/(1 + e−2x)] - 1
  '''

  def __init__(self):
    super().__init__()
    self.x = 0

  def forward(self, input):
    self.x = input
    return (self.x.exp() - (-self.x).exp())/(self.x.exp()+(-self.x).exp())
  
  def backward(self, gradwrtoutput):
    dl_tanh =  self.x.clone().fill_(1) - self.forward(self.x).pow(2)
    return dl_tanh * gradwrtoutput

  def param(self):
    return [(None,None)]

#Module Class: Sequential

class Sequential(Module):
  '''
  Combines several linear and non-linear modules in a Sequential structure (The Multi-Layer Perceptron)
  '''

  def __init__(self, *args):
    super().__init__()
    self.modules = []

    for module in args:
      if(isinstance(module,Module)):
        self.add_module(module)
      else:
        raise ArgumentError("Only modules can be passed as parameters to Sequential Module")
  
  def add_module(self, module):
    self.modules.append(module)
  

  def forward(self, input):
    out = input
    for module in self.modules:
      out = module.forward(out)
    return out

  def backward(self, grdwrtoutput):
    reversed_list = self.modules[::-1]
    grad = grdwrtoutput
    for module in reversed_list:
      grad = module.backward(grad)

  def param(self):
    parameters = []
    for module in self.modules:
      parameters.append(module.param())
    return parameters

#Optimizer Class: SGD

class SGD():
  '''
  Stochastic Gradient Descent Optimizer

  zero_grad() : Clears all the parameter's gradients
  step() : Does one optimization step (Updating the parameters)
  '''

  def __init__(self, params, lr):
    if lr < 0.0 :
      raise ValueError("Invalid learning rate: {} -- [Should be positive]".format(lr))
    
    self.params = params
    self.lr = lr

  def zero_grad(self):
    for module in self.params:
      for pair in module:
        param, dl_param = pair
        if (param is None) or (dl_param is None):
          continue
        else:
          dl_param.zero_()

  def step(self):
    for module in self.params:
      for pair in module:
        param, dl_param = pair
        if (param is None) or (dl_param is None):
          continue
        else:
          param.add_(-self.lr * dl_param)

#Loss Function: MSE

def MSELoss(target, pred):
  '''
  Outputs the MSE Loss as a float value
  '''
  return (pred - target.float()).pow(2).sum()


def dl_MSELoss(target, pred):
  '''
  Outputs gradient's Loss as a Tensor
  '''
  return 2*(pred - target.float())
