"""
Implements all the loss function.
"""
from . import Module
class MSELoss(Module):
    """
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,y_actual,y_pred):
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
        (y_actual - y_pred)

    def backward(self):
        return super().backward()