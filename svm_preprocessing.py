# %%
# Load the raw CIFAR-10 data.
import numpy as np
import matplotlib.pyplot as plt
import data_utils
import download

import os


def load_data_folder():
    '''Function added to load the data'''
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    download_dir = "./data"
    if not os.path.exists(download_dir):
        print("download again")
        download.maybe_download_and_extract(url,download_dir)
    cifar10_dir = 'data/CIFAR-10/cifar-10-batches-py' 
    return cifar10_dir


def train_test_process(X_train,X_test,y_train,y_test):
    # Set the number of samples for each set
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500
    print(len(y_train))

    # Our validation set will be num_validation points from the original
    # training set.
    X_val = X_train[num_training:num_training + num_validation]
    y_val = y_train[num_training:num_training + num_validation]

    # Our training set will be the first num_train points from the original
    # training set.
    X_train = X_train[:num_training]
    y_train = y_train[:num_training]

    # We will also make a development set, which is a small subset of
    # the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our
    # test set.
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)



    '''
    
    # Our validation set will be num_validation points from the original training set.
    X_val = X_train[num_training:num_training + num_validation]
    y_val = y_train[num_training:num_training + num_validation]
  
    # Our training set will be the first num_train points from the original training set.
    X_train = X_train[:num_training]
    y_train = y_train[:num_training]
    
    # We will also make a development set, which is a small subset of the training set.
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # We use the first num_test points of the original test set as our test set.
    X_test = X_test[:num_test]
    y_test = y_test[:num_test]

    # Print shapes to verify the splits
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Development data shape: ', X_dev.shape)
    print('Development labels shape: ', y_dev.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)


    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # As a sanity check, print out the shapes of the data
    print('Training data shape: ', X_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Development data shape: ', X_dev.shape)

    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)
    print(mean_image[:10]) # print a few of the elements
    plt.figure(figsize=(4,4))
    plt.imshow(mean_image.reshape((32,32,3)).astype('uint8')) # visualize the mean image
    plt.show()

    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # third: append the bias dimension of ones (i.e. bias trick) so that our SVM
    # only has to worry about optimizing a single weight matrix W.
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    print(X_train.shape, X_val.shape, X_test.shape, X_dev.shape)
    '''





# %%
