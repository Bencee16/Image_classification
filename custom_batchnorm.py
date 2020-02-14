import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    """
    super(CustomBatchNormAutograd, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps
    self.gamma = nn.Parameter(torch.eye(n_neurons), requires_grad=True)
    self.beta = nn.Parameter(torch.zeros((n_neurons, 1)), requires_grad=True)

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor

    """

    assert(input.shape[1] == self.n_neurons)

    mu = torch.mean(input, 0)
    var = torch.var(input, 0, unbiased=False)  #(1./n_batch) * torch.sum((input - mu)**2, 0)
    x_hat = (input-mu) * torch.rsqrt(var+self.eps)
    out = torch.add(torch.mm(x_hat, self.gamma), self.beta.t())

    return out


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    """

    B = input.shape[0]
    n_neurons = input.shape[1]
    gamma = torch.diag(gamma)

    beta = torch.tensor(torch.unsqueeze(beta,0),requires_grad=False)
    mu = torch.mean(input,0)
    var = torch.var(input,0, unbiased=False)
    inv_var = torch.rsqrt(var+eps)
    x_hat = (input-mu) * inv_var

    res= x_hat @ gamma
    out = torch.add(res, beta)

    ctx.eps = eps
    ctx.B = B
    ctx.save_for_backward(input, gamma, beta, mu, var, inv_var, x_hat)
    # ctx.save_for_backward(input, gamma, beta, mu, var, inv_var, x_hat)

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    """

    input, gamma, beta, mu, var, inv_var, x_hat = ctx.saved_variables
    eps = ctx.eps
    B = ctx.B
    grad_input = grad_gamma = grad_beta = None

    grad_beta = torch.sum(grad_output, 0)
    grad_gamma = torch.sum(torch.mul(grad_output,x_hat),0)

    grad_xhat = torch.mm(grad_output,gamma)
    # grad_mu = torch.sum(-inv_var*grad_xhat, 0)
    # grad_sigma = 0.5 * 1./in_var
    # grad_input = inv_var*grad_xhat + 1./B * grad_mu.repeat(B,1) +

    grad_input = (1. / B) * inv_var * (B * grad_xhat - torch.sum(grad_xhat,0) - x_hat * torch.sum(grad_xhat*x_hat, 0))
    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    """
    super(CustomBatchNormManualModule, self).__init__()

    self.n_neurons = n_neurons
    self.eps = eps
    self.gamma = nn.Parameter(torch.eye(n_neurons),requires_grad=True)
    self.beta = nn.Parameter(torch.zeros((n_neurons,1)),requires_grad=True)


  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    """

    assert(input.shape[1]==self.n_neurons)

    bn = CustomBatchNormManualFunction.apply
    out = bn(input, torch.diag(self.gamma), self.beta.squeeze(), self.eps)

    return out
