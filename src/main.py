import pandas as pd
import numpy as np
import os
from typing import Dict, Tuple
from numba import jit


def generate_columns() -> Dict[str, type]:
    """
    Generates the columns for the DataFrame. The column names are based off
    the variables table at https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm

    Returns:
      Dict[str, type]: A dictionary of column names and their respective types
    """

    columns: dict = {}

    for i in range(1, 50):
        columns[f"X{i}"] = np.float64
        columns[f"Y{i}"] = np.float64

    for i in range(1, 50):
        columns[f"Power{i}"] = np.float64

    columns.update({"qW": np.float64, "Total_Power": np.float64})
    return columns


path = os.path.join("..", "data", "WEC_Perth_49.csv")
df: pd.DataFrame = pd.read_csv(path, dtype=generate_columns())

X = df.drop(["Total_Power"], axis=1)
y = df["Total_Power"]

print(f"X shape: {X.shape} and y shape: {y.shape}")

# Dropping positional columns
X = X.drop(X.columns[X.columns.str.contains("X")], axis=1)
X = X.drop(X.columns[X.columns.str.contains("Y")], axis=1)

print("Removed positional columns")
print(f"X shape: {X.shape} and y shape: {y.shape}")


# Linear Regression
@jit(nopython=True)
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error between the true and predicted values

    Args:
      y_true (np.ndarray): The true values
      y_pred (np.ndarray): The predicted values

    Returns:
      float: The Mean Squared Error
    """

    return (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)


@jit(nopython=True)
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the R^2 score between the true and predicted values

    Args:
      y_true (np.ndarray): The true values
      y_pred (np.ndarray): The predicted values

    Returns:
      float: The R^2 score
    """

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


@jit(nopython=True)
def gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float, epochs: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the optimal weights for a linear regression model using
    gradient descent algorithm.

    Uses the mean squared error as the loss function to minimize.

    Args:
        X (np.ndarray): The input features of the dataset
        y (np.ndarray): The true values of the dataset
        lr (float): The learning rate of the algorithm
        epochs (int): The number of iterations to run the algorithm

    Returns:
        Tuple[np.ndarray, np.ndarray]: The optimal weights of the model and the loss  at each epoch
    """

    m, n = X.shape
    theta: np.ndarray = np.zeros(n)
    loss: np.ndarray = np.zeros(epochs)

    for i in range(epochs):
        y_pred = X.dot(theta)
        loss[i] = mse(y, y_pred)

        gradient: np.ndarray = (1 / m) * X.T.dot((y_pred - y))
        theta -= lr * gradient

    return theta, loss


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset into training, validation, and testing sets based on the
    provided ratios.

    Args:
        X (np.ndarray): The input features of the dataset
        y (np.ndarray): The true values of the dataset
        train_ratio (float, optional): The ratio of the training set.
                                       Defaults  to 0.6.
        val_ratio (float, optional): The ratio of the validation set.
                                     Defaults to 0.2.
        test_ratio (float, optional): The ratio of the testing set.
                                      Defaults to 0.2.
        random_state (int, optional): The random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The training, validation, and testing sets for X and y
    """
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    np.random.seed(random_state)

    n: int = len(y)
    indices: np.ndarray = np.arange(n)
    np.random.shuffle(indices)

    X_shuffled = X[indices]
    y_shuffled = y[indices]

    train_size: int = int(n * train_ratio)
    val_size: int = int(n * val_ratio)

    X_train = X_shuffled[:train_size]
    y_train = y_shuffled[:train_size]

    X_val = X_shuffled[train_size : train_size + val_size]
    y_val = y_shuffled[train_size : train_size + val_size]

    X_test = X_shuffled[train_size + val_size :]
    y_test = y_shuffled[train_size + val_size :]

    return X_train, X_val, X_test, y_train, y_val, y_test


def standard_scaler(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scales the input features of the dataset using the standard scaler.

    Args:
        X_train (np.ndarray): The training set
        X_val (np.ndarray): The validation set
        X_test (np.ndarray): The testing set

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The scaled training, validation, and testing sets
    """

    mean: np.ndarray = np.mean(X_train, axis=0)
    std: np.ndarray = np.std(X_train, axis=0)

    X_train_scaled: np.ndarray = (X_train - mean) / std
    X_val_scaled: np.ndarray = (X_val - mean) / std
    X_test_scaled: np.ndarray = (X_test - mean) / std

    return X_train_scaled, X_val_scaled, X_test_scaled


X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    X.values, y.values, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
)

X_train_scaled, X_val_scaled, X_test_scaled = standard_scaler(X_train, X_val, X_test)

# bias terms
X_train_scaled = np.c_[np.ones(len(X_train_scaled)), X_train_scaled]
X_val_scaled = np.c_[np.ones(len(X_val_scaled)), X_val_scaled]
X_test_scaled = np.c_[np.ones(len(X_test_scaled)), X_test_scaled]

alpha: float = 0.01
epochs: int = 1000

theta = np.zeros(X_train_scaled.shape[1])
theta, loss = gradient_descent(X_train_scaled, y_train, alpha, epochs)

y_train_pred = X_train_scaled.dot(theta)
y_pred_val = X_val_scaled.dot(theta)
y_pred_test = X_test_scaled.dot(theta)

print("\n Results")
print(f"Training R^2: {r2(y_train, y_train_pred)}")
print(f"Validation R^2: {r2(y_val, y_pred_val)}")
print(f"Test R^2: {r2(y_test, y_pred_test)}")

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

ax[0].plot(np.arange(epochs), loss, linewidth=2)
ax[0].set_title("Loss vs Epochs", fontsize=15)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

sns.scatterplot(x=y_test, y=y_pred_test, alpha=0.7, s=50, ax=ax[1])
sns.lineplot(x=y_test, y=y_test, color="red", ax=ax[1])
ax[1].set_title("True vs Predicted Values")
ax[1].set_xlabel("True Values")
ax[1].set_ylabel("Predicted Values")

plt.show()
