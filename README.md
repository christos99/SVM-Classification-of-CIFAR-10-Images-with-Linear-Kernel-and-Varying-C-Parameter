## SVM Image Classification using CIFAR-10 Dataset

This project demonstrates the implementation of a Support Vector Machine (SVM) for image classification using the CIFAR-10 dataset. The code trains an SVM model with a linear kernel and evaluates its performance on the CIFAR-10 test set.This repository contains Python code for image classification using Support Vector Machines (SVM) with different kernel functions. The code is divided into three separate files, each focusing on a specific kernel type: linear, polynomial, and sigmoid.

### Dataset Description

The CIFAR-10 dataset consists of 60,000 color images in 10 different classes, with 6,000 images per class. The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each image has a size of 32x32 pixels.

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- numpy
- matplotlib
- scikit-learn
- keras

You will also need the CIFAR-10 dataset, which can be loaded using the `cifar10` module from `keras.datasets`.

## Project Structure

The code is organized into three separate files:

1. `linear_svm.py`: Implements an SVM classifier with a linear kernel. It loads the CIFAR-10 dataset, preprocesses the data, and performs SVM classification using different `c` values. It calculates the accuracy on the testing and training sets and displays the training time. It also plots example predictions for `c=0.1`.

2. `polynomial_svm.py`: Implements an SVM classifier with a polynomial kernel. It follows a similar structure as `linear_svm.py` but uses a polynomial kernel instead. It evaluates the SVM model's performance with different `c` values and plots example predictions for `c=0.1`.

3. `sigmoid_svm.py`: Implements an SVM classifier with a sigmoid kernel. It has the same structure as the previous files but uses a sigmoid kernel. It evaluates the SVM model's performance with different `c` values and plots example predictions for `c=0.1`.

## Code Structure

The code follows a similar structure in all three files, with the only difference being the kernel function used. Each file consists of the following sections:

1. Importing Dependencies: The necessary libraries and modules are imported, including numpy, matplotlib, scikit-learn, keras, and specific modules for SVM and metrics.

2. Dataset Loading and Preprocessing: The CIFAR-10 dataset is loaded using the `cifar10.load_data()` function. The dataset is then preprocessed by converting the data type to float, reshaping the images into vectors, and normalizing the pixel values to a range of -1 to 1.

3. Dataset Partitioning: The code selects a subset of the dataset for training and testing purposes. This step is done to reduce the computational time, and you can adjust the number of samples used for training and testing by modifying the code accordingly.

4. Model Training and Evaluation: The SVM model is instantiated with the specified kernel type (linear, polynomial, or sigmoid) and trained using the training data. The model's accuracy is evaluated on both the training and testing sets using the `accuracy_score` metric. The training time is also measured.

5. Result Visualization: The code includes visualizations to illustrate the model's performance. For example, when `c=0.1`, it plots four false predicted images and four correctly predicted images.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


