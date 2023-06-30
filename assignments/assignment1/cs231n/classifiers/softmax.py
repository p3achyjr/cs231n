from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    dW = np.zeros_like(W) # (D, C)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.mean(scores)
      correct_class_score = scores[y[i]]

      exp_scores = np.exp(scores)
      log_exp_scores = np.log(np.sum(exp_scores))
      loss += -correct_class_score + log_exp_scores

      for j in range(num_classes):
        if j == y[i]:
          dW[:, j] -= X[i]
      
        dW[:, j] += (exp_scores[j] / np.sum(exp_scores) * X[i])

    loss /= num_train
    dW /= num_train 

    loss += reg * np.sum(W * W)
    dW += reg * 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.

    - W: (D, C)
    - X: (N, D)
    - y: (N,)
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    d = W.shape[0]
    c = W.shape[1]

    XW = np.matmul(X, W) # (N, C), all masses
    XW -= np.mean(XW)

    XExp = np.exp(XW) # (N, C), exponentiated mass

    fY = XW[np.arange(n), y] # (N,). mass at correct label
    fExpSum = np.sum(XExp, axis=1) # (N,). total label mass

    L = np.log(fExpSum) - fY
    loss = np.sum(L) / n + reg * np.sum(W * W)

    dWy_mask = np.zeros_like(XW) # (N, C)
    dWy_mask[np.arange(n), y] = 1 # (N, C). dWy_mask[i, y[i]] = 1
    dWy = np.matmul(np.transpose(X), dWy_mask) # (D, C), correct label mask

    XExp_normalized = XExp / np.expand_dims(fExpSum, axis=1)
    dWj = np.matmul(np.transpose(X), XExp_normalized) # (D, C). summed exponentiated mass.
    dW = (dWj - dWy) / n
    dW += reg * 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
