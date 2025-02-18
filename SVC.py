# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:52:28 2025

@author: asiva
"""
#%% Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
#%% Define dataset parameters
n_samples = 800  # Number of data points
n_features = 2   # Number of features (input dimensions)
n_classes = 2    # Number of classes (binary classification)

# Generate a synthetic dataset with two classes
X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=1,
                            n_redundant=0, n_repeated=0, n_clusters_per_class=1,
                            n_classes=n_classes, random_state=42)

# Convert class labels from 0 to -1 (SVM typically uses -1 and 1 labels)
y = np.where(y == 0, -1, y)

# Visualize the dataset
plt.figure(figsize=(8,6))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1')
plt.title('Linear Classifier Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
#%% Standardize the dataset (feature scaling)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
#%% Define the SVM classifier class
class SVM(object):
    def __init__(self, lambda_parameter=0.001, learning_rate=0.001, epochs=1000):
        self.learning_rate = learning_rate  # Step size for weight updates
        self.lambda_parameter = lambda_parameter  # Regularization parameter
        self.epochs = epochs  # Number of training iterations
        self.weight = None  # Model weights
        self.bias = None  # Bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weight = np.zeros(n_features)  # Initialize weights with zeros
        self.bias = 0  # Initialize bias as zero

        # Training loop for a given number of epochs
        for _ in range(self.epochs):
            for i in range(n_samples):
                predicted_value = y[i] * (np.dot(self.weight, X[i]) + self.bias)
                
                # If the sample is misclassified, update weights and bias
                if predicted_value < 1:
                    self.weight -= self.learning_rate * (self.lambda_parameter * self.weight - y[i] * X[i])
                    self.bias -= self.learning_rate * (-y[i])
                
                # Otherwise, update weights only (regularization term)
                else:
                    self.weight -= self.learning_rate * (self.lambda_parameter * self.weight)

        return self.weight, self.bias

    def predict(self, X):
        # Compute the prediction using the learned weights and bias
        return np.sign(np.dot(self.weight, X) + self.bias)
#%% Train the SVM classifier
svm_classifier = SVM(lambda_parameter=0.001, learning_rate=0.01, epochs=1000)
w, b = svm_classifier.fit(X, y)
#%% Generate x1 values for plotting the decision boundary
x1_values = np.linspace(min(X[:,0]), max(X[:, 0]), len(X))
#%% Compute x2 values for the decision boundary
w1 = w[0]
w2 = w[1]

if w2 != 0:
    x2_values = (-w1 * x1_values - b) / w2  # Solve for x2 in decision boundary equation
#%% Plot the dataset and decision boundary
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='b', label='Class 1', edgecolor='k')
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], c='r', label='Class -1', edgecolor='k')
plt.plot(x1_values, x2_values, c='g', linestyle='--', label='Decision Boundary')

# Labels and formatting
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Classifier for Linearly Separable Data')
plt.legend()
plt.grid(True)
plt.show()
#%%