import numpy as np
from utils.utils import to_categorical
from utils.utils import load_dataset
from utils.optimizer import SGD
from train import train_sgd
from model import NeuralNetwork
import matplotlib.pyplot as plt
import argparse 

if __name__ ==  "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--path_dataset',type='str',default='./dataset')
    args.parse_args()
    np.random.seed(100)
    
    x_train,y_train,x_test,y_test = load_dataset(args.path_dataset)
    
    print(f'Training Examples: {x_train.shape}')
    print(f'Training Labels:   {y_train.shape}')
    print(f'Testing Examples:  {x_test.shape}')
    print(f'Testing Labels:    {y_test.shape}')
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    x_train/=255
    x_test/=255
    
    
    print(f'Training Examples: {x_train.shape}')
    print(f'Training Labels:   {y_train.shape}')
    print(f'Testing Examples:  {x_test.shape}')
    print(f'Testing Labels:    {y_test.shape}')


    
    #Hyper parameters.
    learning_rate = 0.01
    batch_size = 64
    epochs = 100
    #Declare Model.
    
    input_dim = x_train.shape[1]
    neurons_per_layer = [128,64,10]
    number_of_layer = len(neurons_per_layer)
    net = NeuralNetwork(number_of_layer,input_dim,neurons_per_layer,'Relu')
    sgd = SGD(net,learning_rate)
    net.print_model()
    #
    trained_model = train_sgd(net,x_train,x_test,y_train,y_test,sgd,batch_size,epochs)
    