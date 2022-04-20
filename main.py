
from network.layer import LinearLayer
from network.activations import Sigmoid
from network.loss import MSELoss
import numpy as np

if __name__ ==  "__main__":
    x = np.random.rand(100,100)
    y = np.random.uniform(1,10,size=(100,1)) 
    l1 = LinearLayer(100,1)
    
    z = l1(x)
    act = Sigmoid()
    a = act(z)
    loss = MSELoss()
    l = loss.forward(y,a)
    print(loss.grad)