from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    dW = np.transpose(dW)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    dW_j = np.zeros(dW.shape)
    dW_y = np.zeros(dW.shape)
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        num_above_margin = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                num_above_margin += 1
                dW_j[j] += X[i]
        dW_y[y[i]] -= num_above_margin * X[i]
         #print("naive: ", i, num_above_margin, y[i], dW_y[y[i]])

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    reg_loss = reg * np.sum(W * W)
    loss += reg_loss

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW = (np.transpose(dW_j + dW_y)) / num_train + 2*W*reg

    # print("naive dW_j:", dW_j)
    # print("naive dW_y:", dW_y)
    # print("naive dW:", dW)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.

    - W: (D, C)
    - X: (N, D)
    - y: (N,)
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    n = X.shape[0]
    d = W.shape[0]
    c = W.shape[1]

    WT = np.transpose(W) # (C, D)
    XT = np.transpose(X) # (D, N)

    Wx_j = np.matmul(WT, XT) # (C, N)
    Wx_yi = Wx_j[y, np.arange(0, n)] # (1, N)

    Wdiff = np.maximum(Wx_j - Wx_yi + 1, np.zeros_like(Wx_j)) # (C, N)
    Wdiff[y, np.arange(0, n)] = 0
    loss_f = np.sum(Wdiff) / n
    loss_reg = reg * np.sum(W * W)

    loss = loss_f + loss_reg

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    Wdiff_counts = (Wdiff > 0).astype(np.uint8) # (C, N)
    Wy_counts = np.expand_dims(np.sum(Wdiff_counts, axis=0), axis=1) # (1, N)
    
    dW_j = np.matmul(Wdiff_counts, X) # (N, D)

    y_indicators = np.zeros((c, n))
    y_indicators[y, np.arange(n)] = 1
    # print("y_indicators", y_indicators)
    dW_y = Wy_counts * X # (N, D)
    # print("dW_y", dW_y.shape, dW_y)
    dW_y = np.matmul(y_indicators, dW_y)

    # print("dW_y", dW_y.shape, dW_y)
    dW = np.transpose(dW_j - dW_y) / n + 2*W*reg

    # print("Wdiff_counts", Wdiff_counts)
    # print("Wy_counts", Wy_counts)
    # print("dW_j", dW_j)
    # print("dW_y", dW_y)
    # print("vectorized:", dW)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
