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
    if isinstance(error_vec, np.float64):
        error_vec = np.array([error_vec])
    return error_vec

#####################
# SUM SQUARED ERROR #
######################
def sse_func(X, Y, W, b_mat):
    """Compute sum squared error."""
    errors_vec = raw_error(X, Y, W, b_mat)
    err_squared = np.square(errors_vec)
    sum_squared_error = sum(err_squared)
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
    rmse_gradient = mse_gradient * (1/(2*rmse))
    return rmse_gradient


####################
# REGULARIZED SSE  #
####################
def reg_sse_func(X, Y, W, l, b_mat):
    """Compute the regularized SSE with given lambda parameter `l`."""
    sse = sse_func(X, Y, W, b_mat)
    reg_sse = sse + (l/2) * np.linalg.norm(W) ** 2
    return reg_sse


def reg_sse_grad(X, Y, W, l, b_mat):
    """Compute the gradient of the regularized SSE w/ lambda `l`."""
    sse_gradient = sse_grad(X, Y, W, b_mat)
    reg_term_gradient = 

########################
# OPTIMIZATION METHODS #
########################
def batch_gradient_descent(X, Y, M=2, mu=0.01, err_func=sse_func, err_grad_func=sse_grad, record_fitting=False):
    W = np.zeros((M,))
    b_mat = gen_basis_matrix(X, M)

    prev_score = float("inf")
    converge_thresh = 1.99
    converge_counter = 0
    training_record = []
    i = 0

    while converge_counter < 10 and i < 1500:
        # Update weights
        err_grad = err_grad_func(X, Y, W, b_mat)
        update = mu*err_grad
        W = W - update

        # Evaluate current weights
        score = err_func(X, Y, W, b_mat)
        score_diff = abs(score - prev_score)
        prev_score = score
        if score < converge_thresh:
            converge_counter += 1
        else:
            converge_counter = 0
        # print(i, W, score, score_diff)
        
        training_record.append((i, W, score))
        if record_fitting and i%5 == 0:
            approx_vec = approximation(X, Y, W, b_mat)
            rec_fig, rec_ax = plt.subplots(1,1)
            rec_ax.scatter(X, Y, c='blue')
            rec_ax.scatter(X, approx_vec, c='red', marker='x')
            rec_fig.savefig(f"./training_images/batch/{i}.png")
            plt.close()
        
        i+= 1
    
    if record_fitting and M == 2:
        W0 = []
        W1 = []
        scores = []

        for r in training_record:
            w0, w1 = r[1]
            score = r[2]
            W0.append(w0)
            W1.append(w1)
            scores.append(score)

        batch_converge_fig = plt.figure()
        ax = Axes3D(batch_converge_fig)
        ax.scatter(W0, W1, scores, c=range(len(W0)))
        batch_converge_fig.savefig("./training_images/batch_covergence_plot.png")

    
    return W, score, i, training_record


def stoch_gradient_descent(X, Y, M=2, mu=0.01, err_func=sse_func, err_grad_func=sse_grad, record_fitting=False):
    W = np.zeros((M,))
    b_mat = gen_basis_matrix(X, M)

    prev_score = float("inf")
    converge_thresh = 1.99
    converge_counter = 0
    training_record = []
    i = 0

    while converge_counter < 10 and i < 2500:
        n = i % len(X)
        x = np.array([X[n]])
        y = np.array([Y[n]])
        b_vec = b_mat[n]
        err_grad = err_grad_func(x, y, W, b_vec)
        update = mu*err_grad
        W = W - update
        
        # Evaluate current weights
        score = err_func(X, Y, W, b_mat)
        score_diff = abs(score - prev_score)
        prev_score = score
        if score < converge_thresh:
            converge_counter += 1
        else:
            converge_counter = 0
        # print(i, W, score, score_diff)
        
        training_record.append((i, W, score))
        if record_fitting and i%5 == 0:
            approx_vec = approximation(X, Y, W, b_mat)
            rec_fig, rec_ax = plt.subplots(1,1)
            rec_ax.scatter(X, Y, c='blue')
            rec_ax.scatter(X, approx_vec, c='red', marker='x')
            rec_fig.savefig(f"./training_images/stoch/{i}.png")
            plt.close()
        
        i+= 1
    
    if record_fitting and M == 2:
        W0 = []
        W1 = []
        scores = []

        for r in training_record:
            w0, w1 = r[1]
            score = r[2]
            W0.append(w0)
            W1.append(w1)
            scores.append(score)

        batch_converge_fig = plt.figure()
        ax = Axes3D(batch_converge_fig)
        ax.scatter(W0, W1, scores, c=range(len(W0)))
        batch_converge_fig.savefig("./training_images/stoch_covergence_plot.png")

    
    return W, score, i, training_record


def newton_method(X, Y, M=2, mu=0.01, err_func=sse_func, err_grad_func=sse_grad, record_fitting=False):
    W = np.zeros((M,))
    b_mat = Phi_X = gen_basis_matrix(X, M)
    lin_regress_hessian = np.matmul(Phi_X.T, Phi_X)
    lin_reg_hess_inv = np.linalg.inv(lin_regress_hessian)
    
    prev_score = float("inf")
    converge_thresh = 1.99
    converge_counter = 0
    training_record = []
    i = 0

    while converge_counter < 10 and i < 1500:
        update = np.dot(lin_reg_hess_inv, err_grad_func(X, Y, W, b_mat))
        W = W - mu*update
        
        # Evaluate current weights
        score = err_func(X, Y, W, b_mat)
        score_diff = abs(score - prev_score)
        prev_score = score
        if score_diff < 10**-6:
            converge_counter += 1
        else:
            converge_counter = 0
        # print(i, W, score, score_diff)

        training_record.append((i, W, score))
        if record_fitting and i%5 == 0:
            approx_vec = approximation(X, Y, W, b_mat)
            rec_fig, rec_ax = plt.subplots(1,1)
            rec_ax.scatter(X, Y, c='blue')
            rec_ax.scatter(X, approx_vec, c='red', marker='x')
            rec_fig.savefig(f"./training_images/newton/{i}.png")
            plt.close()
        
        i+= 1

    if record_fitting and M == 2:
        W0 = []
        W1 = []
        scores = []

        for r in training_record:
            w0, w1 = r[1]
            score = r[2]
            W0.append(w0)
            W1.append(w1)
            scores.append(score)

        batch_converge_fig = plt.figure()
        ax = Axes3D(batch_converge_fig)
        ax.scatter(W0, W1, scores, c=range(len(W0)))
        batch_converge_fig.savefig("./training_images/newton_covergence_plot.png")
    
    return W, score, i, training_record
    

    
if __name__ == "__main__":
    M = 2  # Will train M features
    x_train, y_train, x_test, y_test = load_data(viz=False)
    b_W, b_score, b_iters, b_record = batch_gradient_descent(x_train, y_train, M, 
                                                            err_func=sse_func,
                                                            err_grad_func=sse_grad,
                                                            record_fitting=True)
    s_W, s_score, s_iters, s_record = stoch_gradient_descent(x_train, y_train, M, 
                                                            err_func=sse_func,
                                                            err_grad_func=sse_grad,
                                                            record_fitting=True)
    n_W, n_score, n_iters, n_record = newton_method(x_train, y_train, M=2, mu=0.1,
                                                    err_func=sse_func,
                                                    err_grad_func=sse_grad,
                                                    record_fitting=True)

    print("Summary:")
    print(f"Batch GD: Weights={b_W} | Final Score: {b_score} | Iters: {b_iters}")
    print(f"Stochastic GD: Weights={s_W} | Final Score: {s_score} | Iters: {s_iters}")
    print(f"Newton's Method: Weights={n_W} | Final Score: {n_score} | Iters: {n_iters}")

    converge_fig = plt.figure()
    converge_ax = Axes3D(converge_fig)
    for ri, record in enumerate([b_record, s_record, n_record]):
        W0 = []
        W1 = []
        scores = []

        for r in record:
            w0, w1 = r[1]
            score = r[2]
            W0.append(w0)
            W1.append(w1)
            scores.append(score)
        cmap_str = {0: 'Blues',
                    1: 'Greens',
                    2: 'Reds'}[ri]
        converge_ax.scatter(W0, W1, score, cmap=cmap_str, c=range(len(W0)))
    converge_fig.savefig("./convergences.png")

    weights_dict = {}
    training_scores = []
    for m in range(0, 10):
        weights, score, i, training_record = newton_method(x_train, y_train, M=m+1, mu=0.1, err_func=rmse_func, err_grad_func=rmse_grad)
        # weights, score, i, training_record = batch_gradient_descent(x_train, y_train, M=m+1, err_func=sse_func, err_grad_func=sse_grad, record_fitting=True)
        weights_dict[m] = weights
        training_scores.append(score)
        print(score)

    test_scores = []
    for m in range(0, 10):
        objective_score = rmse_func(x_test, y_test, weights_dict[m], gen_basis_matrix(x_test, M=m+1))
        test_scores.append(objective_score)
        print(objective_score)
        
    fig, ax = plt.subplots(1,1)
    ax.plot(training_scores, label="Training", c="blue")
    ax.scatter(range(0, 10), training_scores, c="blue")
    ax.plot(test_scores, label="Test", c="red")
    ax.scatter(range(0, 10), test_scores, c="red")
    fig.legend(loc=(0.45, 0.75)) 
    fig.savefig("./overfitting.png")
    plt.close()