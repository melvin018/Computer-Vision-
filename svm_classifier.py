from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import matplotlib.pyplot as plt

def svm_loss_naive(W, X, y, reg):

    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[0]
    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y]  
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1) 
    margins[np.arange(num_train), y] = 0  
    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W) 

    binary = margins > 0
    binary = binary.astype(int)
    incorrect_counts = np.sum(binary, axis=1)  
    binary[np.arange(num_train), y] = -incorrect_counts  
    dW = X.T.dot(binary) / num_train
    dW += 2 * reg * W  

    return loss, dW



