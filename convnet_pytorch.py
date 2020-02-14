"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    """

    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.batch_norm1 = nn.BatchNorm2d(64)
    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.batch_norm2 = nn.BatchNorm2d(128)
    self.conv3_a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.batch_norm3_a = nn.BatchNorm2d(256)
    self.conv3_b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.batch_norm3_b = nn.BatchNorm2d(256)

    self.conv4_a = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
    self.batch_norm4_a = nn.BatchNorm2d(512)
    self.conv4_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batch_norm4_b = nn.BatchNorm2d(512)

    self.conv5_a = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batch_norm5_a = nn.BatchNorm2d(512)
    self.conv5_b = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
    self.batch_norm5_b = nn.BatchNorm2d(512)

    self.avg_pool = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    self.linear = nn.Linear(512, n_classes)

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    x = F.max_pool2d(F.relu(self.batch_norm1(self.conv1(x))), 3, stride=2, padding=1)
    x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 3, stride=2, padding=1)
    x = F.max_pool2d(F.relu(self.batch_norm3_b(self.conv3_b(F.relu(self.batch_norm3_a(self.conv3_a(x)))))), 3, stride=2, padding=1)
    x = F.max_pool2d(F.relu(self.batch_norm4_b(self.conv4_b(F.relu(self.batch_norm4_a(self.conv4_a(x)))))), 3, stride=2, padding=1)
    x = F.max_pool2d(F.relu(self.batch_norm5_b(self.conv5_b(F.relu(self.batch_norm5_a(self.conv5_a(x)))))), 3, stride=2, padding=1)
    x = self.avg_pool(x).view(-1,512)
    out = self.linear(x)

    ########################
    # END OF YOUR CODE    #
    #######################

    return out
