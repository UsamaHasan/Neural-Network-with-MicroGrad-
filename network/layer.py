from . import Module
import numpy as np
"""

""" 
class LinearLayer(Module):
    """
    """
    def __init__(self,input,output) -> None:
        """
        """
        super().__init__()
        self.w = np.random.rand(output,input)
        self.b = np.random.rand(output,1)

    def forward(self,x)-> np.array:
        """
        Implements the forward pass of Linear Layer.
        .. math::
            w * x + b
        Parameters
        ----------
        x: np.array
            array containing input.
        Returns
        -------
        return: array
            
        """
        return np.dot(self.w,x) + self.b
    
    def backward(self):
        pass

class ConvLayer(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    pass