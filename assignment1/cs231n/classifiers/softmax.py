import numpy as np
from random import shuffle


#implementation from https://github.com/haofeixu/standford-cs231n-2018/blob/master/assignment1/softmax.ipynb

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]

  dscores = np.zeros((num_train, num_classes))

  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    p = np.exp(scores) / np.sum(np.exp(scores))

    one_hot_y = np.zeros(num_classes)
    one_hot_y[y[i]] = 1
    dscores[i] = p - one_hot_y

    loss += -np.log(p[y[i]])

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)

  dscores /= num_train
  dW = X.T.dot(dscores)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  num_train=X.shape[0]
  f=X.dot(W) # N x C
  f-=f.max()
  f=np.exp(f)
  F=f/f.sum(axis=-1).reshape(-1,1) # N x C
  p=F[np.arange(num_train),y] # N x 1
  loss = np.sum(-np.log(p))/num_train
  loss += reg * np.sum(W*W)
  F[np.arange(num_train),y]-=1
  dW = X.T.dot(F)
  dW /= num_train
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

