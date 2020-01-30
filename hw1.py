from math import log, sqrt

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_data(viz=True):
    # Load data
    x_test = np.load('data/q2xTest.npy')
    x_train = np.load('data/q2xTrain.npy')
    y_test = np.load('data/q2yTest.npy')
    y_train = np.load('data/q2yTrain.npy')

    if viz:
    # Visualize training and testing data
        raw_data_fig, raw_data_ax = plt.subplots(1,1)
        raw_data_ax.scatter(x_train, y_train, c='blue', label='Training')
        raw_data_ax.scatter(x_test, y_test, c='red', label='Testing')
        raw_data_fig.legend(loc=(0.7, 0.75))
        plt.show()
        raw_data_fig.savefig("./raw_data.png")

    return x_train, y_train, x_test, y_test


def gen_basis_matrix(data, M):
    """Pre-compute a basis matrix with M features."""
    basis_matrix = np.zeros(shape=(len(data), M))
    for di, d in enumerate(data):
        for m in range(M):
            basis_matrix[di, m] = d ** m
    return basis_matrix


def approximation(X, Y, W, b_mat=None):
    if b_mat is None:
        b_mat = gen_basis_matrix(X, M=len(W))
    approx_vec = np.dot(b_mat, W)
    return approx_vec


def raw_error(X, Y, W, b_mat=None):
    """Compute the raw error for each data point with the given weights."""
    if b_mat is None:
        b_mat = gen_basis_matrix(X, M=len(W))
    approx_vec = approximation(X, Y, W, b_mat)
    error_vec = approx_vec - Y
    return raw_error

#####################
# SUM SQUARED ERROR #
######################
def sse_func(X, Y, W, b_mat):
    """Compute sum squared error."""
    errors_vec = raw_error(X, Y, W, b_mat)
    sum_squared_error = sum(errors_vec) ** 2
    adjusted_sse = 0.5 * sum_squared_error
    return adjusted_sse


def sse_grad(X, Y, W, b_mat):
    """Compute the gradient wrt W of the sse."""
    error_vec = raw_error(X, Y, W, b_mat)
    error_vec = error_vec.reshape(error_vec.shape + (1, ))
    gradient_mat = np.multiply(error_vec, b_mat)
    gradient_vec = np.sum(gradient_mat, axis=0)
    return gradient_vec

######################
# MEAN SQUARED ERROR #
######################
def mse_func(X, Y, W, b_mat):
    """Compute the mean squared error."""
    sse = sse_func(X, Y, W, b_mat)
    mse = sse / len(X)
    return mse


def mse_grad(X, Y, W, b_mat):
    """Compute the gradient wrt W of the mse."""
    sse_gradient = sse_grad(X, Y, W, b_mat)
    mse_gradient = sse_gradient / len(X)
    return mse_gradient


##########################
# ROOT MEAN SQUARE ERROR #
##########################
def rmse_func(X, Y, W, b_mat):
    """Compute the root mean-square-error."""
    mse = mse_func(X, Y, W, b_mat)
    rmse = sqrt(mse)
    return rmse


def rmse_grad(X, Y, W, b_mat):
    rmse = rmse_func(X, Y, W, b_mat)
    mse_gradient = mse_grad(X, Y, W, b_mat)
    
    

    
if __name__ == "__main__":
    M = 2  # Will train M features
    x_train, y_train, x_test, y_test = load_data()
    x_train_basis_mat = gen_basis_matrix(x_train, M=M)