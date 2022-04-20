
from network.layer import LinearLayer
from network.activations import Sigmoid
from network.loss import MSELoss , CrossEntropyLoss
import numpy as np

if __name__ ==  "__main__":
    x = np.random.rand(100,10)
    y = np.random.uniform(1,10,size=(10,1)) 
    l1 = LinearLayer(100,1)
    
    z = l1(x)
    act = Sigmoid()
    a = act(z)
    loss = CrossEntropyLoss()
    l = loss.forward(a,y)
    