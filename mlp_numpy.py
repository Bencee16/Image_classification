"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP

    """

    self.layers = {}

    self.num_layers = len(n_hidden)+1

    if self.num_layers==1:
      self.layers['linear1'] = LinearModule(n_inputs, n_classes)
      self.layers['softmax'] = SoftMaxModule()

    else:
      self.layers['linear1'] = LinearModule(n_inputs, n_hidden[0])
      self.layers['relu1'] = ReLUModule()
      for i in range(1, self.num_layers-1):
        self.layers['linear'+str(i+1)] = LinearModule(n_hidden[i-1], n_hidden[i])
        self.layers['relu'+str(i+1)] = ReLUModule()

      self.layers['linear'+str(self.num_layers)] = LinearModule(n_hidden[-1], n_classes)
      self.layers['softmax'] = SoftMaxModule()


  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network

    """


    for i in range(self.num_layers-1):
      x = self.layers['relu'+str(i+1)].forward(self.layers['linear'+str(i+1)].forward(x))

    out = self.layers['softmax'].forward(self.layers['linear'+str(self.num_layers)].forward(x))


    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss

    """

    dx = self.layers['linear'+str(self.num_layers)].backward(dout)

    for i in range(self.num_layers-1, 0, -1):
      dx = self.layers['linear'+str(i)].backward(self.layers['relu'+str(i)].backward(dx))

    return
