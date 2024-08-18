import pandas as pd
import numpy as np
from typing import Dict


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


df: pd.DataFrame = pd.read_csv("./WEC_Perth_49.csv", dtype=generate_columns())

X: pd.Series = df.drop(["Total_Power"], axis=1)
y: pd.Series = df["Total_Power"]

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler: StandardScaler = StandardScaler()
features_scaled: np.ndarray = scaler.fit_transform(X)

pca: PCA = PCA(random_state=42)
pca_result: np.ndarray = pca.fit_transform(features_scaled)

cumulative_variance_ratio: np.ndarray = np.cumsum(pca.explained_variance_ratio_)
n_components_95: int = np.argmax(cumulative_variance_ratio >= 0.95) + 1

df_pca: pd.DataFrame = pd.DataFrame(
    pca_result[:, :n_components_95],
    columns=[f"PC{i+1}" for i in range(n_components_95)],
)

df_pca["Total_Power"] = y

X_pca = df_pca.drop(["Total_Power"], axis=1)
y_pca = df_pca["Total_Power"]

from numba import jit
from typing import Tuple


@jit(nopython=True)
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the mean squared error between the true and predicted values
    of a dataset.

    Args:
        y_true (np.ndarray): The true values of the dataset
        y_pred (np.ndarray): The predicted values of the dataset

    Returns:
        float: The mean squared error between the true and predicted values
    """

    return (1 / len(y_true)) * np.sum((y_true - y_pred) ** 2)


@jit(nopython=True)
def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calculates the R^2 score between the true and predicted values of a dataset.

    Args:
        y_true (np.ndarray): The true values of the dataset
        y_pred (np.ndarray): The predicted values of the dataset

    Returns:
        float: The R^2 score between the true and predicted values
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


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_pca, test_size=0.2, random_state=42, shuffle=True
)

X_train_gd: np.ndarray = np.c_[np.ones(X_train.shape[0]), X_train]
y_train_gd: np.ndarray = y_train.values

alpha: float = 0.003
epochs: int = 2000

theta, loss = gradient_descent(X_train_gd, y_train_gd, alpha, epochs)
y_pred_gd: np.ndarray = X_test.dot(theta[1:]) + theta[0]
mse_gd: float = mse(y_test.to_numpy(), y_pred_gd.to_numpy())
r2_gd: float = r2(y_test.to_numpy(), y_pred_gd.to_numpy())

print(f"Mean Squared Error: {mse_gd:.4f}")
print(f"R^2 Score: {r2_gd:.4f}")

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = [24, 8]

fig, ax = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle("PCA and Linear Regression Analysis", fontsize=24)

# Loss curve
sns.lineplot(x=range(len(loss)), y=loss, ax=ax[0])
ax[0].set_title("Loss vs. Epochs", fontsize=16)
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Loss")

# Actual vs predicted total power (gradient descent)
sns.scatterplot(x=y_test, y=y_pred_gd, ax=ax[1])
ax[1].plot([y.min(), y.max()], [y.min(), y.max()], "k--", lw=4)
ax[1].set_title("Actual vs Predicted (Gradient Descent)", fontsize=16)
ax[1].set_xlabel("Actual Total Power")
ax[1].set_ylabel("Predicted Total Power")

# Residuals plot
residuals_gd = y_test - y_pred_gd

sns.scatterplot(x=y_pred_gd, y=residuals_gd, ax=ax[2])
ax[2].axhline(y=0, color="r", linestyle="--")
ax[2].set_title("Residuals Plot (Gradient Descent)", fontsize=16)
ax[2].set_xlabel("Predicted Total Power")
ax[2].set_ylabel("Residuals")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
