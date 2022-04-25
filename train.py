from model import NeuralNetwork
from tqdm import tqdm
import numpy as np
from utils.dataset import Batch_scheduler
from utils.optimizer import SGD

def train_sgd(net:NeuralNetwork,x_train:np.array,x_test:np.array,y_train:np.array\
    ,y_test:np.array,sgd:SGD,batch_size:int,epochs:int) -> NeuralNetwork:
    """
    Parameters
    ----------
    net: NeuralNetwork
        Model object to be trained
    x_train: np.array
        Input Training set.
    x_test: np.array
        Input testing set.
    y_train: np.array
        Training Label set.
    y_test: np.array
        Testing Label set.
    learning_rate : int
        Learning rate 
    batch_size: int
        Train example size per batch
    epochs: int
        Total number of training Iterations
    Returns
    -------
    return : NeuralNetwork
        Trained model object.
    """
    #split the dataset too.
    
    train_loader = Batch_scheduler(x_train,y_train,batch_size)
    test_loader = Batch_scheduler(x_test,y_test,batch_size)
    for iter in tqdm(range(epochs)):
        training_loss = 0.0
        for mini_batch in train_loader:
            x , y = mini_batch
            output = net(x)
            training_loss+=net.criterion(y,output)
            net.backward(output,y)
            sgd.step()
        print(f'Epoch:{iter}: Loss{training_loss/len(train_loader)}')
    
    eval_loss = 0.0    
    for mini_batch in test_loader:
        x , y = mini_batch
        output = net(x)
        loss = net.criterion(y,output)
            
            
   

