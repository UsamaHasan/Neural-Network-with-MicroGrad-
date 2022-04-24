
from model import NeuralNetwork
import numpy as np
from network import activations
from utils.utils import to_categorical
if __name__ ==  "__main__":
    
    network = NeuralNetwork(3,100,[20,10,3])
    x=np.random.rand(5,100)
    y = np.random.randint(0,3,(5,1))
    y = to_categorical(y)
    y_hat=network(x)
    network.print_model()
    loss = network.criterion(y,y_hat)
    #print(y_hat.shape)
    network.backward(y_hat,y)
    