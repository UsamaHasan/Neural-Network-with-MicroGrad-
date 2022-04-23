"""
Implements all the loss function.
"""
from . import Module
import numpy as np
class MSELoss(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,y_actual:np.array,y_pred:np.array) -> np.array:
        """
        Calculates the Mean Squared Loss.
        -- math::
            1/N * sum_{(y_actual - y_pred)^2}
        Parameters
        ----------
        y_actual: np.array
            Ground Truth values/Labels
        y_pred: np.array
            Predicted values/Labels 
        Returns
        -------
        return: np.array
            mse loss vector.
        """
        N = y_actual.shape[0]
        l = 1/N *  np.sum(y_actual - y_pred)**2
        return l

    def backward(self):
        return super().backward()



class BCELoss(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        return super().forward()

    def backward(self):
        return super().backward()

class CrossEntropyLoss(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    def forward(self,x,y):
        """
        Calculates Logits and then log-likelihood estimation
        .. math::
            y_hat = exp(x)/sum_{exp(x_i)}
            loss = -sum_{y_i*log(yhat_i)}
        Parameters
        ----------
        x: np.array
            Model output
        y: np.array
            Actual Label.
        Return
        ------
        return: float
            Loss based on cross entropy.
        """
        N = y.shape[0]
        y_hat = np.exp(x)/ np.sum(np.exp(x))
        loss = -np.sum(np.dot(y,np.log(y_hat))) / N
        return loss

    def backward(self,input,output):
        """
        Backpropagates
        .. math::
            derivative of e^x = e^x
            derivative  of log(s) = 1/s*derivative(s) 
            derivative of softmax = softmax*(1-softmax)
            derivative of cross_entropy = y_hat - y
        """
        
        return output - input