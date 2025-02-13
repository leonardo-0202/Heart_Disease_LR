# Organized imports
import numpy as np
import matplotlib.pyplot as plt
import costs as cst

# We will have to organise files before uploading the project
import helpers as help

from typing import Tuple

# I think the print boolean parameter does not break the tests because it is optional, for us to use only
def mean_squared_error_gd(y: np.ndarray, 
                          tx: np.ndarray,
                          initial_w: np.ndarray,
                          max_iters: int,
                          gamma: float,
                          print = False) -> Tuple[np.ndarray, float]:
    """Computes the weights and the corresponding loss after some fixed number of iterations

    Args:
        y: shape=(N, ).
        tx: shape=(N,D).
        initial_w: shape=(D, ). The initial vector of weights.
        max_iters: shape = (). Max. number of iterations.
        gamma: shape = (). learning rate.

    Returns:
        An array w of shape (d, ), containing the last weight vector obtained, 
        the loss calculated at w of shape ().
        """
    # Initalization of w for first iteration
    w = initial_w
    for n_iter in range(max_iters):
        # Compute gradient
        e = y - tx @ w
        g = -1/y.shape[0] * np.transpose(tx) @ e
        
        # update w
        w = w - gamma * g

        ######### Set flag true to show trace, this could be removed to avoid checking the if we wanta bit more speed
        if print:
            loss = cst.compute_loss_MSE(y,tx,w)
            print(
                "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return w, cst.compute_loss_MSE(y,tx,w)

def mean_squared_error_sgd(y: np.ndarray,
                           tx: np.ndarray, 
                           initial_w: np.ndarray, 
                           max_iters: int,
                           gamma: float) -> Tuple[np.ndarray, float]:
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        An array w of shape (d, ), containing the last weight vector obtained, 
        the loss calculated at w of shape ().
    """
    # Initalization of w for first iteration
    w = initial_w
    for n_iter in range(max_iters):
        # Store the gradient of each minibatch (we will use only one in this case)
        b_gradients = []
        for minibatch_y, minibatch_tx in help.batch_iter(y, tx, 1):
            # Compute gradient
            e = y - tx @ np.transpose(w)
            g = -1/y.shape[0] * np.transpose(tx) @ e
            
            #TODO?: since we only do one iter, we can change the lists to a sinlge value for speed
            b_gradients.append(g)

        # Compute the sum of the gradient of each minibatch
        g = sum(b_gradients)/len(b_gradients)

        # Update w
        w = w - gamma*g
        
        loss = cst.compute_loss_MSE(y,tx,w)
        print(
            "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return w, cst.compute_loss_MSE(y,tx,w)
  
def least_squares(y: np.ndarray, 
                  tx: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    """
    
    w = np.linalg.solve(tx.T @ tx ,tx.T @ y )
    mse = 1 / (2 * y.shape[0]) * ((y - tx @ w) ** 2).sum()
    
    return w,mse

def ridge_regression(y: np.ndarray, 
                     tx: np.ndarray, 
                     lambda_: float) -> Tuple[np.ndarray, float]:
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """

    w = np.linalg.solve(tx.T @ tx +  2 * y.shape[0] * lambda_ * np.eye(tx.shape[1]),tx.T @ y )
    
    return w

def logistic_regression(y: np.ndarray, 
                        tx: np.ndarray, 
                        initial_w: np.ndarray, 
                        max_iters: int, 
                        gamma: float) -> Tuple[np.ndarray, float]:
    
    """implement logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N,1)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D,1), for each iteration of GD

    """
    #retieving the length of y
    N = y.shape[0]
    #initialising the guess of the model parameter
    ws = [initial_w]
    losses = []
    w = initial_w
    #defining a the sigma function
    def sigmoid(t):
        return 1/(1 + np.exp(-t))

    for n_iter in range(max_iters):
        # computing the loss and the gradient at each iteration
        linear_terms = tx @ w
        log_likelihood = np.log(1 + np.exp(-y * linear_terms))
        loss = np.mean(log_likelihood)
        
        g  = 1 / N * (tx.T @ (sigmoid(tx @ w) - y))
        # computing w at the step t+1
        w = w - gamma * g
        # saving the computed w at the iteration t+1
        ws.append(w)
        losses.append(loss)
    
    return losses,ws


def reg_logistic_regression(y: np.ndarray, 
                            tx: np.ndarray, 
                            initial_w: np.ndarray, 
                            max_iters: int, 
                            gamma: float, 
                            lambda_: float) -> Tuple[np.ndarray, float]:
    
    """implement regularized logistic regression using gradient descent.

     Args:
        y: numpy array of shape=(N,1)
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(D,1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (D,1), for each iteration of GD

    """
    #retieving the length of y
    N = y.shape[0]
    #initialising the guess of the model parameter
    ws = [initial_w]
    losses = []
    w = initial_w
    #defining a the sigma function
    def sigmoid(t):
        return 1/(1 + np.exp(-t))
    
    for n_iter in range(max_iters):
        # computing the loss and the gradient at each iteration
        linear_terms = tx @ w
        log_likelihood = np.log(1 + np.exp(-y * linear_terms))
        loss = np.mean(log_likelihood) + (lambda_ / 2) * np.sum(w ** 2)
                
        g = 1 / N * (tx.T @ (sigmoid(tx @ w) - y)) +  lambda_ * w
        # computing w at the step t+1
        w = w - gamma * g
        # saving the computed w at the iteration t+1
        ws.append(w.copy())
        losses.append(loss) 
    
    return losses,ws