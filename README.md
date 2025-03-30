# Support Vector Machine (SVM) Classifier

## Overview

This repository contains an implementation of a **Support Vector Machine (SVM) classifier** for binary classification using Python. The classifier is implemented from scratch and trained on a synthetically generated dataset. The model learns to separate two classes using a linear decision boundary.

## Features

- Generates a **synthetic dataset** for binary classification using `make_classification`.
- **Standardizes** the dataset using `StandardScaler`.
- Implements **SVM from scratch** using gradient descent.
- Supports **customizable hyperparameters** such as learning rate, regularization parameter, and number of epochs.
- Visualizes the dataset and **decision boundary**.

## Dataset

The dataset is generated using the `make_classification` function from `sklearn.datasets`. It consists of:

- **800 samples**
- **2 features** (one informative feature, no redundant features)
- **2 classes** (binary classification)
- Data is standardized using `StandardScaler` for better optimization performance.

## Model Implementation

The SVM model is implemented from scratch and trained using gradient descent with a **hinge loss function**:

### Objective Function

The loss function used is:
L = \lambda \|w\|^2 + \sum_{i=1}^{n} \max(0, 1 - y_i (w \cdot x_i + b))

Where:

- **w**: Weight vector
- **b**: Bias term
- **y**: Target labels (-1 or 1)
- **x**: Input features
- **\lambda**: Regularization parameter

### Training Algorithm

1. Initialize weights `w` and bias `b` to zero.
2. Iterate over **epochs**:
   - For each sample:
     - Compute the predicted value.
     - Update weights and bias using **stochastic gradient descent** (SGD) if the sample is misclassified.
     - Apply regularization.
3. Train until convergence.

## Dependencies

To run this project, you need the following Python libraries:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

Clone the repository and run the script:

```bash
git clone https://github.com/Abhi05-goat/SVM_From_Scratch.git
cd SVM_From_Scratch
python SVC.py
```

## Visualization

The script generates a **scatter plot** of the dataset with the learned **decision boundary**:

- Red points represent **Class -1**.
- Blue points represent **Class 1**.
- Green dashed line represents the **decision boundary**.

## Hyperparameters

You can modify the hyperparameters in the `SVM` class:

```python
svm_classifier = SVM(lambda_parameter=0.001, learning_rate=0.01, epochs=1000)
```

- `lambda_parameter`: Regularization parameter (default: `0.001`)
- `learning_rate`: Step size for weight updates (default: `0.01`)
- `epochs`: Number of training iterations (default: `1000`)

## Results

The classifier successfully learns a **linear decision boundary** that separates the two classes effectively, achieving high accuracy on linearly separable data.

## Author

[Abhivanth] - [https://github.com/Abhi05-goat](https://github.com/Abhi05-goat)

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

Feel free to contribute by submitting pull requests or reporting issues!
