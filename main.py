
from network.layer import LinearLayer
from network.activations import Sigmoid
import numpy as np

if __name__ ==  "__main__":
    a = np.random.rand(100,1)
    l1 = LinearLayer(100,10)
    act = Sigmoid()
    b = act(a)