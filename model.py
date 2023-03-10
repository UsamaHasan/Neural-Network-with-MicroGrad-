from network import Module
import numpy as np
from network.activations import *
from network.layer import LinearLayer
from network.loss import CrossEntropyLoss
class NeuralNetwork(Module):
    """
    """
    def __init__(self,number_of_layer:int,input_dim:int,neurons_per_layer: list,activation ='Sigmoid') -> None:
        """
        Parameters
        ----------
        number_of_layer: int
            Number of Hidden layers.
        input_dim: int
            Input Dimension
        neurons_per_layer: list
            Number of neurons per layers.
        """
        super().__init__()
        assert number_of_layer == len(neurons_per_layer)
        self.module_list = []
        self.error_function = CrossEntropyLoss()
        for i,n in enumerate(neurons_per_layer):
            self.module_list.append(LinearLayer(input_dim,n))
            if i + 1 < len(neurons_per_layer):
                self.module_list.append(eval(activation)())
            input_dim = n

    def forward(self,x) -> np.array:
        """
        """
        for layer in self.module_list:
            x = layer(x)
        return x
    def backward(self,output,gt) -> np.array:
        """
        BackPropagates through the network
        Parameters
        ----------
        output: np.array
            Output Of model.
        gt: np.array
            Ground Truth Labels.
        """
        grads = self.error_function.backward(output,gt)
        
        for layer in reversed(self.module_list):
            grads = layer.backward(grads)
        
    def print_model(self) -> None:
        """
        """
        for layer in self.module_list:
            print(layer)

    def criterion(self,y:np.array,y_hat:np.array)->np.array:
        """
        Parameters
        ----------
        y: np.array
            Actual Label of data.
        y_hat: np.array
            Predicted Labels of data.
        Returns
        """
        error = self.error_function.forward(y_hat,y)
        return error

