"""
"""
import numpy as np

class Tensor():
    """
    Implements datatype tensor a placeholder that stores numpy ndarray and 
    a computation graph to perform computation
    """
    def __init__(self,arr:np.array) -> None:
        """
        Parameters
        ----------
        arr: np.array
            Input array of tensor
        """
        self.value = arr
    def __str__(self) -> str:
        return f'Tensor:{self.value}'
    def __repr__(self) -> str:
        return f'Tensor:{type(self.value)}'