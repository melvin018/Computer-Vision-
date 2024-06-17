from svm_preprocessing import load_data_folder,train_test_process
from data_utils import load_CIFAR10
import numpy as np

load_data_folder()
root = './data/CIFAR-10/cifar-10-batches-py'
X_train,y_train,X_test,y_test = load_CIFAR10(root)
X_dev,y_dev,X_val,y_val = train_test_process(X_train,y_train,X_test,y_test)

