import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings that occur during data conversion in sklearn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def load_and_preprocess_data():
    """
    Load and preprocess the CIFAR-10 dataset.
    Normalize and reshape the data for SVM classification.
    Returns a subset of the data for training and testing to increase efficiency.
    """
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the data to be within the range [-1, 1]
    x_train, x_test = x_train.astype(np.float32) / 127.5 - 1, x_test.astype(np.float32) / 127.5 - 1

    # Flatten the images into vectors for SVM classification
    x_train, x_test = x_train.reshape((x_train.shape[0], -1)), x_test.reshape((x_test.shape[0], -1))

    # Flatten the labels
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Return a smaller subset of the data for quick training and testing
    return x_train[:3000], y_train[:3000], x_test[:2000], y_test[:2000]

def plot_images(images, labels, predictions, class_names):
    """
    Plot a selection of images showing their true labels and predicted labels.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        # Reshape and display the image
        ax.imshow(images[i].reshape(32, 32, 3))

        # Set title with true and predicted labels
        title = f"True: {class_names[labels[i]]}\nPredicted: {class_names[predictions[i]]}"
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def train_and_evaluate_svm(x_train, y_train, x_test, y_test, C, kernel_type):
    """
    Train and evaluate an SVM classifier with a specified kernel type and C value.
    Print the accuracy on the training and testing sets, and the training time.
    Returns the trained classifier.
    """
    print(f"Training SVM with {kernel_type} kernel (C={C})")
    start_time = time.time()

    # Initialize and train the SVM classifier
    clf = SVC(C=C, kernel=kernel_type)
    clf.fit(x_train, y_train)

    # Calculate the time taken to train the classifier
    training_time = time.time() - start_time

    # Evaluate the classifier
    train_accuracy = accuracy_score(y_train, clf.predict(x_train))
    test_accuracy = accuracy_score(y_test, clf.predict(x_test))

    # Print out the results
    print(f"C={C}: Test Acc={test_accuracy:.4f}, Train Acc={train_accuracy:.4f}, Training Time={training_time:.2f}s")
    return clf

def main():
    """
    Main function to execute the SVM training and evaluation process.
    Iterates over different kernels and C values for comparison.
    """
    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # List of kernels and C values to iterate over
    kernels = ['linear', 'poly', 'sigmoid']
    C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    # Iterate over each kernel and C value, training and evaluating the SVM classifier
    for kernel in kernels:
        for C in C_values:
            clf = train_and_evaluate_svm(x_train, y_train, x_test, y_test, C, kernel)

            # For C = 0.1, plot image predictions to visually assess performance
            if C == 0.1:
                predictions = clf.predict(x_test)
                errors = np.where(predictions != y_test)[0]
                corrects = np.where(predictions == y_test)[0]
                sample_indices = np.concatenate([errors[:4], corrects[:4]])
                plot_images(x_test[sample_indices], y_test[sample_indices], predictions[sample_indices], class_names)

if __name__ == "__main__":
    main()
