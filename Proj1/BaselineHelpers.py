import torch
import dlc_practical_prologue as prologue
import torch.nn as nn
import torch.nn.functional as F
import statistics
import matplotlib.pyplot as plt
from models import *

def train_model(model, train_input, train_target ,test_input, test_target, lr = 0.001 , nb_epochs = 25 , batch_size = 1):
    '''
    Function that trains a Baseline model using CrossEntropy as loss function and SGD as optimizer 
    For best perormance we use a batch_size = 1 and learning_rate(=lr) = 0.001
    Returns losses : tensor that contains the train loss per epoch
            test_accuracy :  tensor that contains test accuracy in % per epoch
            train_accuracy : tensor that contains train accuracy in % per epoch
    '''
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = lr) #Adam gave really bad results around 50%
    test_accuracy = torch.zeros(nb_epochs) #
    train_accuracy = torch.zeros(nb_epochs) # tensor that contains train accuracy in % per epoch
    losses = torch.zeros(nb_epochs) # cumulates the train loss per epoch
    for e in range(nb_epochs):
        cumul_loss = 0
        for input , target in zip(train_input.split(batch_size), train_target.split(batch_size)):
            output = model(input)
            loss = criterion(output , target)
            cumul_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_accuracy[e] = (1- compute_nb_errors(model, test_input, test_target) / test_input.size(0) )* 100 
        train_accuracy[e] = (1- compute_nb_errors(model, train_input, train_target) / train_input.size(0) )* 100 
        losses[e] = cumul_loss
    return  losses , test_accuracy , train_accuracy

def compute_nb_errors(model, data_input, data_target , batch_size = 100):
    '''
    Function that returns number of misclassified inputs of the actual input parameters.
    To be called in each epoch to see the evolution of the Baseline model's accuracy
    '''
    nb_errors = 0
    for b in range(0,data_input.size(0) , batch_size):
        output = model(data_input.narrow(0,b, batch_size))
        _ , pred = output.max(1) 
        for k in range(batch_size):
            if data_target[b + k] != pred[k] : nb_errors += 1
    return nb_errors


def test_model(nb_rounds = 10 , nb_epochs = 25):
    '''
    This function will run 10 rounds of random data and weight initialization of the Baseline model
    In each round it will train the randomly generated model's parameter with the random generadet data_input
    Returns : losses         :tensor that contains the train loss per epoch per round
              test_accuracy  :tensor that contains test accuracy in % per epoch per round
              train_accuracy :tensor that contains train accuracy in % per epoch per round
    These tensors will be later used in the visualisation part           
    '''
    test_accuracy = torch.zeros(nb_rounds,nb_epochs)
    train_accuracy = torch.zeros(nb_rounds,nb_epochs)
    losses = torch.zeros(nb_rounds,nb_epochs)
    for round in range(nb_rounds):
        # Generate new data for each new round
        print("generating new model weights and data randomly")
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
        # Initialize the model parameters for each new round
        model = Baseline()
        print("Round " + str(round+1) + "/" + str(nb_rounds) + " in progress")
        loss , test_acc , train_acc = train_model(model, train_input, train_target, test_input, test_target , nb_epochs = nb_epochs )
        test_accuracy[round] = test_acc
        train_accuracy[round] = train_acc
        losses[round] = loss
        print("end round " + str(round+1))
    return losses , test_accuracy , train_accuracy

def visualisation(x , y_mean , y_std , y_label):
    '''
    A helper function that will be called in BaseLine_visualisation
    '''
    plt.plot(x, y_mean , label = 'BaseLine')
    plt.xlabel('Epoch' , fontsize=30)
    plt.ylabel(y_label , fontsize=30)  
    plt.grid(axis='y')
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha = 0.3)
    plt.title('BaseLine ' +y_label + ' per epoch' , fontsize=35 )
    plt.tick_params(axis='both',  labelsize=25)
    plt.legend(fontsize=30)

def BaseLine_visualisation(accuracy , loss , nb_epochs = 25):
    '''
    Function that plots the train accuracy mean and the loss mean over the 10 rounds.
    We will alsor colour the area around the curve to better visualise the model variance. 
    '''
    ep = torch.arange(nb_epochs).numpy()
    loss_std = loss.std(0).detach().numpy()
    loss_mean = loss.mean(0).detach().numpy()
    acc_std = accuracy.std(0).detach().numpy()
    acc_mean = accuracy.mean(0).detach().numpy()
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    visualisation(ep , acc_mean , acc_std , 'test accuracy')
    plt.subplot(1, 2, 2)
    visualisation(ep , loss_mean , loss_std , 'loss')
    plt.show()
  