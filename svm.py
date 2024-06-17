from svm_preprocessing import load_data_folder,train_test_process
from data_utils import load_CIFAR10
from svm_classifier import svm_loss_naive
import numpy as np



load_data_folder()
root = './data/CIFAR-10/cifar-10-batches-py'
X_train,y_train,X_test,y_test = load_CIFAR10(root)
X_dev,y_dev,X_val,y_val = train_test_process(X_train,y_train,X_test,y_test)

## SVM classifier

# generate a random SVM weight matrix of small numbers
W = np.random.randn(3073, 10) * 0.0001 

#compute gradient and loss using svm_loss_naive() function
loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
print('loss: %f' % (loss, ))

