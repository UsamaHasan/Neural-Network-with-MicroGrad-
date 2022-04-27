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
        l = 1/N*np.sum(y_actual - y_pred)**2
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
        return loss
        """
        exps = np.exp(x - np.max(x))
        logits = exps / exps.sum()        
        loss = -np.mean(y * np.log(logits + 1e-8))
        return loss

    def backward(self,y_hat,y):
        """
        Backpropagates
        .. math::
            derivative of e^x = e^x
            derivative  of log(s) = 1/s*derivative(s) 
            derivative of softmax = softmax*(1-softmax)
            derivative of cross_entropy = y_hat - y
        Parameters
        ----------
        y_hat: np.array
            Input from the previous layer or last layer
        y: np.array
            Actual Label of dataset, ground truhts.
        Returns
        -------
        return :np.array
            gradients of cross entropy loss function
        """
        N = y.shape[0]
        exps = np.exp(y_hat - np.max(y_hat))
        y_hat = exps / np.sum(exps)
        self.grad = (y_hat - y)/ N
        return self.grad