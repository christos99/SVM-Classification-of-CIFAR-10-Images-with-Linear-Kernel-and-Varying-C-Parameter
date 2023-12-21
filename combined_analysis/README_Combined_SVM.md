
# Combined SVM Classifier Script

## Overview
This repository contains a single Python script `combined_svm_classifier_with_c_values.py` that integrates the functionality of three previously separate SVM classifier scripts. Each of these scripts originally implemented a Support Vector Machine (SVM) classifier for image classification using the CIFAR-10 dataset, differing only in the kernel type (linear, polynomial, sigmoid).

## Why Combine the Scripts?
The decision to combine these scripts into one was driven by several key factors:

1. **Reduced Redundancy**: The original scripts shared a significant amount of code for loading and preprocessing the dataset, and evaluating the models. Combining them eliminates this redundancy.

2. **Easier Maintenance and Updates**: With a single script, any changes or updates (such as preprocessing steps or dataset handling) only need to be made in one place.

3. **Simplified Experimentation**: Users can now easily compare the performance of different kernels and `C` values within one script, making it easier to experiment and analyze results.

4. **Enhanced Readability**: The combined script is well-organized and commented, making it easier to understand and modify.

## Functionality
The combined script includes the following functionalities:

1. **Data Loading and Preprocessing**: It loads and preprocesses the CIFAR-10 dataset, preparing it for SVM classification.

2. **SVM Training with Different Kernels**: It allows training SVM models with different kernels - linear, polynomial, and sigmoid.

3. **Comparison Across Different `C` Values**: The script includes the capability to train and evaluate the classifiers with different `C` values, which controls the trade-off between smooth decision boundaries and classifying training points correctly.

4. **Accuracy Visualization**: For each kernel type, the script plots the training and testing accuracies across different `C` values, providing a visual comparison of performance.

## Usage
To use the script, simply run it in a Python environment with the necessary dependencies installed. The script will automatically handle the data loading, model training, evaluation, and visualization.

## Dependencies
- numpy
- matplotlib
- keras
- scikit-learn

## Conclusion
This combined script offers a more streamlined, efficient, and user-friendly approach to comparing SVM classifiers with different kernels on the CIFAR-10 dataset.

