"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,cd
              i.e. the average correct predictions over the whole batch
  """

  num_inputs = predictions.shape[0]
  pred_idx  = np.argmax(predictions, axis=1)
  label_idx = np.argmax(targets, axis=1)

  accuracy = sum(pred_idx==label_idx)/num_inputs

  return accuracy


def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  cifar10 = cifar10_utils.get_cifar10('cifar10/cifar-10-batches-py')
  x, y = cifar10['train'].next_batch(FLAGS.batch_size)
  num_inputs = np.prod(x.shape[1:])
  myMLP = MLP(num_inputs, dnn_hidden_units, 10)
  loss_criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(myMLP.parameters(), lr=FLAGS.learning_rate, weight_decay=0.25)

  accuracies = {'train': [], 'test': []}
  loss_curve = {'train': [], 'test': []}

  for j in range(FLAGS.max_steps):

    x = torch.from_numpy(x).contiguous().view(FLAGS.batch_size, -1)
    y = torch.from_numpy(y).contiguous().view(FLAGS.batch_size, -1)
    optimizer.zero_grad()
    outputs = myMLP(x)
    loss = loss_criterion(outputs, torch.argmax(y, 1))
    # print(loss)
    loss.backward()
    optimizer.step()



    if j% FLAGS.eval_freq == 0:

      accuracies['train'].append(accuracy(outputs.detach().numpy(), y.detach().numpy()))
      loss_curve['train'].append(loss.detach().numpy())

      x, y = cifar10['test'].images, cifar10['test'].labels
      x = torch.from_numpy(x).contiguous().view(x.shape[0], -1)
      y = torch.from_numpy(y).contiguous().view(y.shape[0], -1)
      outputs = myMLP(x)
      loss= loss_criterion(outputs, torch.argmax(y,1))
      print(accuracy(outputs.detach().numpy(), y.detach().numpy()))
      accuracies['test'].append(accuracy(outputs.detach().numpy(), y.detach().numpy()))
      loss_curve['test'].append(loss.detach().numpy())



    x, y = cifar10['train'].next_batch(FLAGS.batch_size)


  # After training we evaluate once more on train and test sets
  with torch.no_grad():
    x = torch.from_numpy(x).contiguous().view(FLAGS.batch_size, -1)
    y = torch.from_numpy(y).contiguous().view(FLAGS.batch_size, -1)
    accuracies['train'].append(accuracy(myMLP(x).detach().numpy(), y.detach().numpy()))
    loss_curve['train'].append(loss.detach().numpy())

    x, y = cifar10['test'].images, cifar10['test'].labels
    x = torch.from_numpy(x).contiguous().view(x.shape[0], -1)
    y = torch.from_numpy(y).contiguous().view(y.shape[0], -1)
    outputs = myMLP(x)
    loss = loss_criterion(outputs, torch.argmax(y, 1))
    print(accuracy(outputs.detach().numpy(), y.detach().numpy()))
    accuracies['test'].append(accuracy(outputs.detach().numpy(), y.detach().numpy()))
    loss_curve['test'].append(loss.detach().numpy())
    plot_results(accuracies, loss_curve)


def plot_results(accuracies, loss_curve):

  plt.plot([j for j in range(0, FLAGS.max_steps+1, FLAGS.eval_freq)], accuracies['test'], label='test')
  plt.legend()
  plt.plot([j for j in range(0, FLAGS.max_steps + 1, FLAGS.eval_freq)], accuracies['train'], 'r', label='train')
  plt.legend()
  plt.xlabel('Number of iterations')
  plt.ylabel('accuracy')
  plt.show()



  plt.plot([j for j in range(0, FLAGS.max_steps + 1, FLAGS.eval_freq)], loss_curve['test'], label='test_loss')
  plt.ylim(0, 2)
  plt.legend()
  plt.plot([j for j in range(0, FLAGS.max_steps + 1, FLAGS.eval_freq)], loss_curve['train'], 'r', label='train_loss')
  plt.legend()
  plt.show()


def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()