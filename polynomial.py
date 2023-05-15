import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
%matplotlib inline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Class names for CIFAR-10
classesName = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Convert pixel values to float
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

# Reshape data into a vector and normalize (-1 to 1)
x_train = np.reshape(x_train, (x_train.shape[0], -1))
y_train = np.reshape(y_train, (y_train.shape[0], -1))
x_test = np.reshape(x_test, (x_test.shape[0], -1))
y_test = np.reshape(y_test, (y_test.shape[0], -1))

x_train = ((x_train / 255) * 2) - 1
x_test = ((x_test / 255) * 2) - 1

# Select a smaller dataset (3000 samples for training and 2000 samples for testing)
x_train = x_train[:3000, :]
y_train = y_train[:3000, :]
x_test = x_test[:2000, :]
y_test = y_test[:2000, :]

y_test = np.array(np.reshape(y_test, 2000))

# Function for plotting images
def plt_img(x, ax):
    n_row = 32
    n_col = 32
    n_colour = 3
    x_new = x.reshape((n_row, n_col, n_colour))
    ax.imshow(x_new)

# SVM classification with polynomial kernel
def poly_svm(c):
    clf = SVC(C=c, kernel='poly')
    start = time.time()
    clf.fit(x_train, y_train)
    stop = time.time()
    training_time = stop - start
    y_pred_train = clf.predict(x_train)
    y_pred_test = clf.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    print("For c = " + str(c) + ", accuracy on the testing set is: " + str(test_accuracy) +
          ", accuracy on the training set is: " + str(train_accuracy) + " and training time is equal to: " +
          str(training_time) + " seconds")

    # Plot 4 false predicted images and 4 correctly predicted images for c = 0.1
    if c == 0.1:
        print("Some image predictions for c = 0.1: ")
        n = 4
        err = np.where(y_pred_test != y_test)[0]
        right = np.where(y_pred_test == y_test)[0]
        f, axarr = plt.subplots(2, n, figsize=(20, 8))
        for i in range(n):
            e_i = err[i]
            r_i = right[i]
            plt_img(x_test[e_i, :], axarr[0, i])
            title1 = 'true={0:s} est={1:s}'.format(classesName[y_test[e_i].astype(int)],
                                                   classesName[y_pred_test[e_i].astype(int)])
            axarr[0, i].set_title(title1)
            plt_img(x_test[r_i, :], axarr[1, i])
            title2 = 'true={0:s} est={1:s}'.format(classesName[y_test[r_i].astype(int)],
                                                   classesName[y_pred_test[r_i].astype(int)])
            axarr[1, i].set_title(title2)

# Running SVM for different c values with polynomial kernel
c_svm_poly = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in c_svm_poly:
    poly_svm(c)

