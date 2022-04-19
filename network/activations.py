"""
Implements all the activations functions here.
"""

from . import Module
import numpy as np

class Sigmoid(Module):
    """
    """
    def __init__(self) -> None:
        """
        """
        super().__init__()
    def forward(self,x) -> np.array:
        """
        Applies activation function sigmoid on vector input.
        .. math::
            1/(1+e^-x)
        Parameters
        ----------
        x : array
            numpy array, 1D vector        
        Returns
        -------
        return: array
            Output array
        """
        if isinstance(x,np.ndarray):
            return 1 / (1 +  np.exp(-x))
        else:
            raise Exception(f"Invalid Input Format:{type(x)}, required np.array") 
    def backward(self,x) -> np.array:
        """
        Implements the derivative of sigmoid function.
        .. math::
            e^-x/(1+e^-x)^2
        Parameters
        ----------
        x : array
        """
        return np.exp(-x) / (1 +  np.exp(-x)) **2

class Tanh(Module):
    """
    """
    def __init__(self) -> None:
        """
        """
        ...
    def forward(self) -> np.array:
        """
        """
        ...
    def backward(self) -> np.array:
        """
        """
        ... 

class Relu(Module):
    """
    """
    def __init__() -> None:
        """
        """
        ...
    def forward() -> np.array:
        """
        """
        ... 
    def backward() ->np.array:
        ...
