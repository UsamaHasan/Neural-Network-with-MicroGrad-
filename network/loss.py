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

class CrossEntropyLoss(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    def forward():
        """
        """
        pass
    def backward(self):
        return super().backward()