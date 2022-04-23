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
        self.b = np.random.randn(output,1)

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
        if self.trainable == True:
            self.ctx = x
        return np.dot(self.w.T,x) + self.b
    
    def backward(self,grad):
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
        self.grad =  grad*self.ctx
        return self.grad
class ConvLayer(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    pass

