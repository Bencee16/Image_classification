"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    self.cache = {'in_x' : None}
    self.num_inputs = None
    self.params['weight'] = np.random.randn(out_features, in_features) * 0.0001
    self.params['bias'] = np.zeros((1, out_features))
    self.grads['weight'] = np.zeros((out_features, in_features))
    self.grads['bias'] = np.zeros((1,out_features))


  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    self.num_inputs = x.shape[0]
    out = x @ (self.params['weight'].T) + self.params['bias']
    self.cache['in_x'] = x

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module

    """

    self.grads['weight'] = dout.T @ self.cache['in_x']
    self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)
    dx = dout @ self.params['weight']
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def __init__(self):
    self.cache = {'in_x': None}

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    """

    self.cache['in_x'] = x
    out = np.maximum(x, 0)

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    """

    dx = np.copy(dout)
    dx[self.cache['in_x']<0] = 0

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  # def __init__(self):
  #   self.cache = {'in_x': None}

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
   """

    b = x.max(axis = 1, keepdims=True)
    y = np.exp(x-b)
    sum_y = np.sum(y, axis=1, keepdims=True)
    out = y/sum_y

    self.x = out

    return out


  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    """

    x = self.x
    num_examples = x.shape[0]
    num_classes = x.shape[1]

    diag_tensor = np.vsplit(x, num_examples) * np.eye(num_classes)
    outer_tensor = x[:, :, None] * x[:, None, :]
    dx = (dout[:, None,:]@(diag_tensor-outer_tensor)).reshape(num_examples, num_classes)

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    """

    out = -np.mean(np.sum(np.multiply(np.log(x),y),1))

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    """

    batch_size = x.shape[0]
    dx = 1./batch_size * np.multiply(-y, 1./x)

    return dx