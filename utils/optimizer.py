from model import NeuralNetwork
class SGD:
    """
    Implements Stochastic Gradient Decent.
    """
    def __init__(self,net:NeuralNetwork,learning_rate:float) -> None:
        """
        Parameters
        ----------
        net: NeuralNetwork
            Model to update the weights
        learning_rate: float
            Learning rate to update the gradient
        """
        self.net = net
        self.learning_rate =learning_rate
    def step(self)->None:
        """
        .. math::
            w_(i+1) = w_i - learning_rate * dW(i))
            b_(i+1) = b_i - learning_rate * dB(i))
        """
        for layer in self.net.module_list:
            if layer.trainable == True:        
                layer.w-=self.learning_rate*layer.grad
                layer.b-=self.learning_rate*layer.grad_in.sum(axis=0)
        