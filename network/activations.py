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
        self.trainable = False
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
        super().forward(x)
        if isinstance(x,np.ndarray):
            return 1 / (1 +  np.exp(-x))
        else:
            raise Exception(f"Invalid Input Format:{type(x)}, required np.array") 
    def backward(self,grad_in) -> np.array:
        """
        Implements the derivative of sigmoid function.
        .. math::
            e^-x/(1+e^-x)^2
        Parameters
        ----------
        x : array
        """     
        super().backward()
        sig = 1 / (1 +  np.exp(-self.ctx))
        return  (sig*(1-sig)) * grad_in
         
    def __str__(self) -> str:
        return f'Sigmoid Layer'
class Tanh(Module):
    """
    """
    def __init__(self) -> None:
        """
        """
        ...
    def forward(self,x) -> np.array:
        """
        Parameters
        ----------
        x: np.array
            Input
        Return
        ------
        return: np.array
            Squashed output
        """
        return np.tanh(x)
        
    def backward(self) -> np.array:
        """
        """
        ... 

class Relu(Module):
    """
    """
    def __init__(self) -> None:
        """
        """
        super().__init__()
        self.trainable = False
    def forward(self,x) -> np.array:
        """
        """
        return np.maximum(x, 0) 
    def backward(self,x) ->np.array:
        """
        """
        return np.array(x > 0).astype('int')
        
class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x:np.array)->np.array:
        """
        Parameters
        ----------
        x: np.array
            Input array
        Returns
        ------- 
        return: np.array
            probability
        """
        y_hat = np.exp(x)/ np.sum(np.exp(x))
        return y_hat
    def backward(self):
        """
        """
        return super().backward()