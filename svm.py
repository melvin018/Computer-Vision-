from svm_preprocessing import load_data_folder,train_test_process
from data_utils import load_CIFAR10

cifar10_dir = load_data_folder()
X_train,X_test,y_train,y_test = load_CIFAR10(cifar10_dir)
train_test_process(X_train,X_test,y_train,y_test)


