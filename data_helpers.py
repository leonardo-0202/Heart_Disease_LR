import numpy as np
import math

def standardize(x):
    """Stadartize the input data x

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        standartized data, shape=(num_samples, num_features)

    >>> standardize(np.array([[1, 2], [3, 4], [5, 6]]))
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    """

    return ( (x - np.mean(x, axis=0, keepdims=True) ) / 
            (np.std(x, axis = 0, keepdims=True))  ) 

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)
    N = x.shape[0]
    sze = math.floor(N*ratio)

    perm = np.random.permutation(N)

    x_tr = x[perm[0:sze]]
    x_te = x[perm[sze:N]]
    y_tr = y[perm[0:sze]]
    y_te = y[perm[sze:N]]

    return x_tr, x_te, y_tr, y_te

    # ***************************************************

import numpy as np

def build_poly(x, degree):
    """Polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N, D), where N is the number of samples and D is the number of features.
        degree: integer.

    Returns:
        poly: numpy array of shape (N, (D * (degree + 1)))

    >>> build_poly(np.array([[0.0, 1.5], [2.0, 3.0]]), 2)
    array([[ 1.  ,  0.  ,  0.  ,  1.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
           [ 1.  ,  2.  ,  4.  ,  1.  ,  3.  ,  9.  ,  0.  ,  0.  ,  0.  ]])
    """
    # Number of samples (N) and number of features (D)
    N, D = x.shape

    # Create an empty list to hold the polynomial features
    poly_features = []

    # Loop through each feature column
    for d in range(D):
        # Create polynomial features for the d-th feature
        tx = x[:, d]  # Select the d-th feature
        powers = np.arange(degree + 1)  # Create an array of powers from 0 to degree
        # Compute polynomial basis for this feature
        poly_column = tx[:, np.newaxis] ** powers  # Compute tx^j for j=0 to degree
        poly_features.append(poly_column)  # Append the polynomial features

    # Concatenate all polynomial features horizontally
    poly = np.hstack(poly_features)

    return poly

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

# TODO: complete docs
def preprocess_data(degree, x_train, x_test, x_validation):
    x_train_degree = build_poly(x_train, degree)
    x_train_degree = x_train_degree[:, np.std(x_train_degree, axis=0) != 0]
    x_train_degree = standardize(x_train_degree)
    x_train_degree = np.hstack((np.ones((x_train_degree.shape[0], 1)), x_train_degree))
    
    x_test_degree = build_poly(x_test, degree)
    x_test_degree = x_test_degree[:, np.std(x_test_degree, axis=0) != 0]
    x_test_degree = standardize(x_test_degree)
    x_test_degree = np.hstack((np.ones((x_test_degree.shape[0], 1)), x_test_degree))
    
    x_validation_degree = build_poly(x_validation, degree)
    x_validation_degree = x_validation_degree[:, np.std(x_validation_degree, axis=0) != 0]
    x_validation_degree = standardize(x_validation_degree)
    x_validation_degree = np.hstack((np.ones((x_validation_degree.shape[0], 1)), x_validation_degree))

    return [x_train_degree, x_test_degree, x_validation_degree]