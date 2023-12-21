import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Function to load and preprocess the CIFAR-10 dataset
def load_and_preprocess_data():
    '''
    Loads the CIFAR-10 dataset and preprocesses it by normalizing and flattening the images.
    Returns the preprocessed training and testing data.
    '''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Flatten the images
    x_train = x_train.reshape((len(x_train), -1))
    x_test = x_test.reshape((len(x_test), -1))
    return x_train, y_train, x_test, y_test

# Function to train and evaluate SVM classifier for different C values
def train_svm_classifier(x_train, y_train, x_test, y_test, kernel_type, C_values):
    '''
    Trains and evaluates an SVM classifier for a given kernel type and multiple C values.
    Returns a dictionary containing training and testing accuracies for each C value.
    '''
    results = {}
    for C in C_values:
        # Instantiate and train the SVM model
        svm = SVC(kernel=kernel_type, C=C)
        svm.fit(x_train, y_train.ravel())

        # Evaluate the classifier
        train_accuracy = accuracy_score(y_train, svm.predict(x_train))
        test_accuracy = accuracy_score(y_test, svm.predict(x_test))

        results[C] = (train_accuracy, test_accuracy)
    return results

# Function to plot accuracy comparison for different C values
def plot_accuracy_comparison(results, kernel_type):
    '''
    Plots the training and testing accuracies for different C values.
    '''
    C_values = list(results.keys())
    train_accuracies = [results[C][0] for C in C_values]
    test_accuracies = [results[C][1] for C in C_values]

    plt.figure(figsize=(10, 6))
    plt.plot(C_values, train_accuracies, label='Training Accuracy')
    plt.plot(C_values, test_accuracies, label='Testing Accuracy')
    plt.xlabel('C value')
    plt.ylabel('Accuracy')
    plt.xscale('log')
    plt.title(f'Accuracy Comparison for {kernel_type} Kernel')
    plt.legend()
    plt.show()

# Main function to execute the script
def main():
    '''
    Main function to execute the SVM training and comparison for different kernels and C values.
    '''
    # Load and preprocess the CIFAR-10 dataset
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    
    # Define kernels and C values for comparison
    kernels = ['linear', 'poly', 'sigmoid']
    C_values = [0.1, 1, 10, 100]

    # Train and evaluate SVM classifiers for each kernel and C value
    for kernel in kernels:
        print(f"Training and evaluating SVM classifiers with {kernel} kernel...")
        results = train_svm_classifier(x_train, y_train, x_test, y_test, kernel, C_values)
        plot_accuracy_comparison(results, kernel)

if __name__ == '__main__':
    main()
