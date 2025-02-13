"""Some helper functions for project 1."""

import csv
import numpy as np
import os

# Use these as the parameter for usecols parameter in genfromtxt
x_TEST_COL_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 27, 28, 29, 31, 32, 33, 34, 35, 37, 40, 41, 43, 44, 45, 46, 47, 48, 51, 52, 53, 54, 59, 217, 220, 222, 223, 228, 230, 231, 232, 233, 234, 236, 237, 238, 240, 241, 242, 243, 244, 245, 247, 248, 249, 250, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 273, 274, 275, 276, 279, 280, 281, 282, 283, 284, 285, 288, 289, 299, 306, 307, 308, 309, 310, 311, 312, 313, 317, 318]
X_TRAIN_COL_INDEXES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 29, 31, 32, 37, 40, 41, 44, 45, 47, 48, 51, 52, 53, 54, 59, 217, 220, 222, 223, 228, 230, 231, 232, 233, 234, 236, 237, 238, 240, 241, 242, 243, 244, 245, 247, 248, 249, 250, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 273, 274, 275, 276, 279, 280, 281, 282, 283, 284, 285, 288, 289, 299, 306, 307, 308, 309, 310, 311, 312, 313, 317, 318]

def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1,
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def standardize(x):
    """
    Standardize the original data set.

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        x (np.array): training data
        mean_x (np.array): average of the array elements
        std_x (np.array): standard deviation of the flattened array
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

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

def get_valid_cols(data: np.ndarray, nan_proportion: float) -> list[int]:
    """Iterates the dataset columns and returns a numpy array with the indexes of the columns 
    that have a smaller proportion of NaN values than specified and without all values equal.

    Args:
        data: numpy array of shape (X,Y).
        tx: shape=(N,2)
        nan_proportion: float between 0 and 1 indicating the maximum proportion of NaN values in a column.

    Returns:
        A numpy array w of shape (1,N), containing the N indexes of the columns with NaN proportion over
        the threshold and not all values equal.
    """
    # Transpose dataset to iterate over columns
    res = []
    i = 0
    for col in data.T:
        col_nan_prop = float(np.count_nonzero(np.isnan(col))) / float(np.size(col))
        z = np.nan_to_num(col, nan=0)

        if col_nan_prop <= nan_proportion and np.std(z) != 0:
            res.append(i)

        i += 1

    print(i)
    return res

def f1_score(y_true, y_pred):
    
    """Calculate the F1 score, which is the harmonic mean of precision and recall.
    The F1 score is a measure of a test's accuracy. It considers both the precision 
    (the number of true positive results divided by the number of all positive results, 
    including those not identified correctly) and the recall (the number of true positive 
    results divided by the number of positives that should have been retrieved).
    Args:
        y_true (list or array-like): True binary labels in range {0, 1}.
        y_pred (list or array-like): Predicted binary labels in range {0, 1}.
    Returns:
        float: The F1 score, a value between 0 and 1.
    """
    # Calculate True Positives (TP)
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    
    # Calculate False Positives (FP)
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    
    # Calculate False Negatives (FN)
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))
    
    # Calculate Precision
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    
    # Calculate Recall
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
    
    # Calculate F1 Score
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1, precision, recall
