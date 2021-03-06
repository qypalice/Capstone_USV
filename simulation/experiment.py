import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from train import *
from data_loader import produce_dataset, get_dataloaders, dataset_plot
from test_function import *
import numpy as np

if __name__ == '__main__':
    initial_out = sys.stdout
    # create dataset (optional)
    x_range = np.array([2.,2.,np.pi])
    u_range = np.array([1.5,0.5])
    SimLength=10
    Ntraj = 100000
    Ts=0.1
    path = produce_dataset(x_range,u_range,SimLength,Ntraj,Ts)
    sys.stdout = initial_out
    print(path)
    dataset_plot(path,num=500)
    # get parameters
    K = 8
    arg = {
        'encoder':[3,32,64,K],
        'decoder':[K+3,128,64,32,3],
        'hyper':[1.0,3.0,0.3,1e-7,1e-7,1e-7,0.,10]
    }

    epochs = 300
    batch_size = 10
    # get dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(path,batch_size)

    # start training
    file_name = train_the_model(train_loader, val_loader, arg, batch_size, epochs)
    save_model_as_numpy(file_name)
    # get test result
    test_the_model(test_loader, file_name)
    print(file_name)
