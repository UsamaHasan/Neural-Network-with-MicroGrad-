from utils.utils import to_categorical
from utils.utils import load_dataset
from utils.optimizer import SGD
from train import train_sgd
from model import NeuralNetwork

if __name__ ==  "__main__":
    
    x_train,y_train,x_test,y_test = load_dataset('/home/usama/Deep Learning/Assignment_1/Task3_MNIST_Data')
    
    print(f'Training Examples: {x_train.shape}')
    print(f'Training Labels:   {y_train.shape}')
    print(f'Testing Examples:  {x_test.shape}')
    print(f'Testing Labels:    {y_test.shape}')
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])

    print(f'Training Examples: {x_train.shape}')
    print(f'Training Labels:   {y_train.shape}')
    print(f'Testing Examples:  {x_test.shape}')
    print(f'Testing Labels:    {y_test.shape}')

    #Hyper parameters.
    learning_rate = 0.1
    batch_size = 64
    epochs = 100
    #Declare Model.
    number_of_layer = 3
    input_dim = x_train.shape[1]
    neurons_per_layer = [100,100,10]
    
    net = NeuralNetwork(number_of_layer,input_dim,neurons_per_layer)
    sgd = SGD(net,learning_rate)
    #
    trained_model = train_sgd(net,x_train,x_test,y_train,y_test,sgd,batch_size,epochs)