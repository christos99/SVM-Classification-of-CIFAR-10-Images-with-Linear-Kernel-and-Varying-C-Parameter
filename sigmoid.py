import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import warnings
from sklearn.exceptions import DataConversionWarning

# Suppress warnings related to data conversion in sklearn
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

def load_and_preprocess_data():
    """
    Load the CIFAR-10 dataset, normalize and reshape the data.
    Returns a subset of the dataset for quicker training.
    """
    # Load dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize and reshape
    x_train, x_test = x_train.astype(np.float32) / 127.5 - 1, x_test.astype(np.float32) / 127.5 - 1
    x_train, x_test = x_train.reshape((x_train.shape[0], -1)), x_test.reshape((x_test.shape[0], -1))
    y_train, y_test = y_train.flatten(), y_test.flatten()

    # Use a smaller subset for training and testing for efficiency
    return x_train[:3000], y_train[:3000], x_test[:2000], y_test[:2000]

def plot_images(images, labels, predictions, class_names):
    """
    Plot a selection of images with their predicted and true labels.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(32, 32, 3))
        title = f"True: {class_names[labels[i]]}\nPredicted: {class_names[predictions[i]]}"
        ax.set_title(title)
        ax.axis('off')
    plt.show()

def train_and_evaluate_svm(x_train, y_train, x_test, y_test, C, kernel='sigmoid'):
    """
    Train an SVM model with specified kernel and C value.
    Evaluate and print its performance on the training and test sets.
    """
    print(f"Training SVM with {kernel} kernel (C={C})")
    start_time = time.time()
    clf = SVC(C=C, kernel=kernel)
    clf.fit(x_train, y_train)
    training_time = time.time() - start_time

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, clf.predict(x_train))
    test_accuracy = accuracy_score(y_test, clf.predict(x_test))

    print(f"C={C}: Test Acc={test_accuracy:.4f}, Train Acc={train_accuracy:.4f}, Training Time={training_time:.2f}s")

    return clf

def main():
    class_names = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    x_train, y_train, x_test, y_test = load_and_preprocess_data()

    # SVM hyperparameters
    C_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

    for C in C_values:
        clf = train_and_evaluate_svm(x_train, y_train, x_test, y_test, C)

        # Plot sample predictions for C=0.1
        if C == 0.1:
            predictions = clf.predict(x_test)
            errors = np.where(predictions != y_test)[0]
            corrects = np.where(predictions == y_test)[0]
            sample_indices = np.concatenate([errors[:4], corrects[:4]])
            plot_images(x_test[sample_indices], y_test[sample_indices], predictions[sample_indices], class_names)

if __name__ == "__main__":
    main()
