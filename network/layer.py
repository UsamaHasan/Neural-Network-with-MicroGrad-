from . import Module
import numpy as np
"""

""" 
class LinearLayer(Module):
    """
    Implements Multi-Layer preceptron Layer. 
    .. math::
        w * x + b
    """
    def __init__(self,input:np.array,output:np.array) -> None:
        """
        Parameters
        ----------
        input: int
            Input size to MLP.
        output: int
            Output dimension of MLP.
        """
        super().__init__()
        self.w = np.random.randn(input,output)
        self.b = np.random.randn(1,output)

    def forward(self,x:np.array)-> np.array:
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
        return: np.array
            
        """
        super().forward(x)
        return np.dot(x,self.w) 
    
    def backward(self,grad_in:np.array):
        """
        Implements backward pass of Linear Layer, through backward propagation 
        by multiplying gradient of this layer(l) with the gradient of the l+1 
        layer.
        .. math::
            a^l * grad^(l+1)
        Parameters
        ----------
        grad: np.array
        Returns
        """
        super().backward()
        grad_output = np.dot(grad_in, self.w.T)
        self.grad = np.dot(self.ctx.T,grad_in)
        return grad_output

    def __str__(self) -> str:
        return f'Linear Layer: Weight shape: {self.w.shape} Bias shape: {self.b.shape}'

class ConvLayer(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    pass

