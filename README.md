## SVM Image Classification using CIFAR-10 Dataset

This project demonstrates the implementation of a Support Vector Machine (SVM) for image classification using the CIFAR-10 dataset. The code trains an SVM model with a linear kernel and evaluates its performance on the CIFAR-10 test set.

### Dataset Description

The CIFAR-10 dataset consists of 60,000 color images in 10 different classes, with 6,000 images per class. The classes include airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each image has a size of 32x32 pixels.

### Code Overview

The code performs the following steps:

1. Loads the CIFAR-10 dataset using the Keras library's `cifar10` module. The dataset is divided into training and test sets, with 50,000 training images and 10,000 test images.

2. Preprocesses the dataset by converting the pixel values to floating point numbers and reshaping the images into a vector format. The pixel values are then normalized to the range of -1 to 1.

3. Trains an SVM model with a linear kernel on a subset of the training data. The subset contains 3,000 training samples and 2,000 testing samples.

4. Evaluates the trained model on the testing set by calculating the accuracy score.

5. Plots 4 false predicted images and 4 correctly predicted images for the SVM model with a C value of 0.1.

6. Performs SVM classification with different C values (0.0001, 0.001, 0.01, 0.1, 1, 10, 100) and outputs the accuracy on the testing set, accuracy on the training set, and training time for each C value.

The project provides a comprehensive example of using SVM for image classification and serves as a useful reference for working with the CIFAR-10 dataset.

