import numpy as np
def to_categorical(gt):
    """
    Convert Label to one-hot vector. The length of vector is equal to the
    number of classes.
    Parameters
    ----------
    gt: np.array
        Array containing ground truth labels
    Return
    ------
    return : np.array
    """
    size = int(max(gt))
    I = np.identity(size+1)
    hot_vectors = []
    for labels in gt:
        hot_vectors.append(I[labels].squeeze())
    return np.array(hot_vectors)