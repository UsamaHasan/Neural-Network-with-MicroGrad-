"""
"""
import numpy as np

class Tensor():
    """
    Implements datatype tensor a placeholder that stores numpy ndarray and 
    a computation graph to perform computation
    """
    def __init__(self,arr:np.array,ops:str) -> None:
        """
        Parameters
        ----------
        arr: np.array
            Input array of tensor
        ops: str
        """
        self.value = arr
        self.ref = None
    def __str__(self) -> str:
        return f'Tensor:{self.value}'
    def __repr__(self) -> str:
        return f'Tensor:{type(self.value)}'
    def __add__(self,othr):
        return Tensor(self.value + othr.arr,'+')
    def __mul__(self,othr):
        return Tensor(self.value * othr.arr,'*')
    @property
    def ref(self):
        return self.ref
    @ref.setter
    def ref(self,r):
        self._ref = r