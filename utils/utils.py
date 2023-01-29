import numpy as np
import glob
import numpy as np
from typing import Tuple
from matplotlib import image as img
from tqdm import tqdm
import os

def split(arr, chunk_size):
    """
    Split array into chunks
    Parameters
    ----------
    arr: np.array
        Array to be split
    chunk_size: int
        Size of each chunk
    Return
    ------
    return: np.array
        Array of chunks
    """
    for i in range(0, len(arr), chunk_size):
        yield arr[i:i + chunk_size]

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



def load_dataset(path) -> Tuple[np.array,np.array,np.array,np.array]:
    """
    Parameters
    ----------
    path: str
        Path to the dataset folder
    Return
    ------
    return: np.array
        Training Examples
    return: np.array
        Training Labels
    return: np.array
        Testing Examples
    return: np.array
        Tesing Labels
    """
    if path is not None:
        print('Loading Dataset...')
        x_train, y_train, x_test, y_test = [], [], [], []
        train_path = os.path.join(path,'train')
        test_path = os.path.join(path,'test')
        n_classes = len(train_path) 
        
        for i in tqdm(range(n_classes)):
            for filename in glob.glob(os.path.join(os.path.join(train_path , str(i)),'*.png')):
                im=img.imread(filename)
                x_train.append(im)
                y_train.append(i)
        
        for i in tqdm(range(n_classes)):
            for filename in glob.glob(os.path.join(os.path.join(test_path , str(i)),'*.png')):
                im=img.imread(filename)
                x_test.append(im)
                y_test.append(i)
        print('Dataset loaded...')
        return np.array(x_train), np.array(y_train),\
                    np.array(x_test),np.array(y_test)
    else:
        raise Exception(f'Dataset path Not defined')
        
def softmax(x):
    """
    Parameters
    ----------
    x: np.array
        Input array
    Return
    ------
    return: np.array
        Softmax of the input array"""
    exps = np.exp(x - np.max(x))
    y_hat = exps / np.sum(exps)
    return y_hat

def accuracy(y,y_hat):
    """
    Parameters
    ----------
    y: np.array
        Ground_truth
    y_pred: np.array
        Predicted
    Return
    ------
    return: float
    """
    y_hat = softmax(y_hat)
    acc = np.sum(y==y_hat)/y.shape[0]
    return acc