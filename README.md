Overview
This project implements a Support Vector Machine (SVM) classifier from scratch using Python. It includes dataset generation, feature scaling, training, and visualization of the decision boundary. The implementation demonstrates how an SVM can classify linearly separable data using sub-gradient descent.

Features
Custom implementation of an SVM classifier
Uses hinge loss with L2 regularization
Generates a synthetic dataset using make_classification
Standardizes features for better convergence
Visualizes dataset and decision boundary
Requirements
To run this project, you need the following dependencies:

NumPy
Matplotlib
Scikit-learn
You can install them using:

bash
Copy
Edit
pip install numpy matplotlib scikit-learn
Workflow
Dataset Generation

Creates a synthetic dataset with two features and two linearly separable classes.
Labels are converted from {0,1} to {-1,1} for SVM compatibility.
Feature Scaling

Standardizes features using StandardScaler from sklearn.preprocessing.
SVM Training

Implements a simple linear SVM using a sub-gradient descent approach.
Optimizes the hinge loss function with L2 regularization.
Decision Boundary Visualization

Plots the dataset with class labels.
Computes and displays the SVM decision boundary.
How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/svm-classifier.git
cd svm-classifier
Run the script:
bash
Copy
Edit
python svm_classifier.py
Applications
Understanding the inner workings of SVMs
Learning to implement machine learning algorithms from scratch
Experimenting with different hyperparameters for optimization
Future Improvements
Implementing support for non-linearly separable data using kernels
Optimizing training with batch gradient descent
Comparing results with Scikit-learn’s built-in SVM implementation
