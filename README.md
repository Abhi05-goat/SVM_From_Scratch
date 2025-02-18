SVM Classifier from Scratch
Overview
This project implements a Support Vector Machine (SVM) classifier from scratch using Python. It demonstrates how an SVM can classify linearly separable data using sub-gradient descent. The implementation includes dataset generation, feature scaling, model training, and visualization of the decision boundary.

Features
Custom implementation of an SVM classifier
Uses hinge loss with L2 regularization
Generates a synthetic dataset for binary classification
Standardizes features for better model performance
Visualizes the dataset and decision boundary
Workflow
Dataset Generation

Creates a synthetic dataset with two features and two linearly separable classes.
Converts class labels from {0,1} to {-1,1} for SVM compatibility.
Feature Scaling

Standardizes features using StandardScaler for improved convergence.
SVM Training

Implements a simple linear SVM using a sub-gradient descent approach.
Optimizes the hinge loss function with L2 regularization.
Decision Boundary Visualization

Plots the dataset with class labels.
Computes and displays the SVM decision boundary.
Applications
Understanding the inner workings of SVMs
Learning to implement machine learning algorithms from scratch
Experimenting with different hyperparameters for optimization
Future Improvements
Extending support to non-linearly separable data using kernel methods
Optimizing training with batch gradient descent
Comparing results with Scikit-learn’s built-in SVM implementation
