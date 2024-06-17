from svm_preprocessing import load_data_folder
from data_utils import load_CIFAR10

load_data_folder()
X_train,X_test,y_train,y_test = load_CIFAR10()
