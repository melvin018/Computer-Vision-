import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_CIFAR10
from gradient_check import grad_check_sparse
import time

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier.
    """
    cifar10_dir = './data/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    mask_train = list(range(num_training))
    X_train = X_train[mask_train]
    y_train = y_train[mask_train]
    
    mask_val = list(range(num_training, num_training + num_validation))
    if num_validation > 0:
        X_val = X_train[num_training:num_training + num_validation]
        y_val = y_train[num_training:num_training + num_validation]
    else:
        X_val = np.empty((0, X_train.shape[1]))
        y_val = np.empty((0,))
    
    mask_test = list(range(num_test))
    X_test = X_test[mask_test]
    y_test = y_test[mask_test]
    
    mask_dev = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask_dev]
    y_dev = y_train[mask_dev]
    
    # Reshape the data
    if X_train.shape[0] > 0:
        X_train = np.reshape(X_train, (X_train.shape[0], -1))
    if X_val.shape[0] > 0:
        X_val = np.reshape(X_val, (X_val.shape[0], -1))
    if X_test.shape[0] > 0:
        X_test = np.reshape(X_test, (X_test.shape[0], -1))
    if X_dev.shape[0] > 0:
        X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data
    mean_image = np.mean(X_train, axis=0, keepdims=True)
    X_train -= mean_image
    X_val -= mean_image  # Subtract mean image from validation data
    X_test -= mean_image
    X_dev -= mean_image
    
    # Add bias dimension
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev

# Softmax loss function (naive implementation)
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    """
    loss = 0.0
    dW = np.zeros_like(W)

    # Compute the loss and gradients
    scores = X.dot(W)
    num_examples = X.shape[0]
    for i in range(num_examples):
        scores_exp = np.exp(scores[i] - np.max(scores[i]))  # Numerical stability
        scores_exp_sum = np.sum(scores_exp)
        correct_score_exp = scores_exp[y[i]]
        loss += -np.log(correct_score_exp / scores_exp_sum)
        for j in range(W.shape[1]):
            if j == y[i]:
                dW[:, j] += (scores_exp[j] / scores_exp_sum - 1) * X[i]
            else:
                dW[:, j] += (scores_exp[j] / scores_exp_sum) * X[i]

    # Average loss and gradients
    loss /= num_examples
    dW /= num_examples

    # Add regularization to the loss
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW

# Softmax loss function (vectorized implementation)
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version
    """
    loss = 0.0
    dW = np.zeros_like(W)

    # Compute the loss
    scores = X.dot(W)
    num_examples = X.shape[0]
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))  # Numerical stability
    scores_exp_sum = np.sum(scores_exp, axis=1, keepdims=True)
    correct_score_exp = scores_exp[np.arange(num_examples), y]
    loss = np.sum(-np.log(correct_score_exp / scores_exp_sum))
    loss /= num_examples
    loss += reg * np.sum(W * W)

    # Compute the gradients
    coeff = scores_exp / scores_exp_sum
    coeff[np.arange(num_examples), y] -= 1
    dW = X.T.dot(coeff)
    dW /= num_examples
    dW += 2 * reg * W

    return loss, dW

# Predict function
def predict(X, W):
    scores = X.dot(W)
    predicted_labels = np.argmax(scores, axis=1)
    return predicted_labels

# Calculate accuracy
def calculate_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Load CIFAR-10 data
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()

# Generate a random softmax weight matrix and compute the loss
W = np.random.randn(3073, 10) * 0.0001
loss, _ = softmax_loss_naive(W, X_dev, y_dev, 0.0)
print('Naive loss:', loss)

# Calculate accuracy for naive implementation
y_dev_pred = predict(X_dev, W)
accuracy = calculate_accuracy(y_dev, y_dev_pred)
print('Naive accuracy:', accuracy)

# Implement a vectorized version of the softmax loss function
loss, _ = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
print('Vectorized loss:', loss)

# Calculate accuracy for vectorized implementation
y_dev_pred = predict(X_dev, W)
accuracy = calculate_accuracy(y_dev, y_dev_pred)
print('Vectorized accuracy:', accuracy)

# Check the gradient numerically
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, _)
