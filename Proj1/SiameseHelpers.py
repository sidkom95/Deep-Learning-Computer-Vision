import torch
import dlc_practical_prologue as prologue
import torch.nn as nn
import torch.nn.functional as F
import statistics
import matplotlib.pyplot as plt
from models import *

def convert_to_one_hot_labels(input, all_targets , target):
    '''
    This is a modified version of the original convert_to_one_hot_labels in the prologue that suits our implementation.
    But it does the same job
    '''
    tmp = input.new_zeros(input.size(0), all_targets.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp 

def train_siamese_model(model, train_input, train_target ,train_classes,test_input, test_target, nb_epochs = 20 ,  w_sharing = True , aux_loss = True , alpha = 0.2, lr = 0.001 , batch_size = 1 ):
    '''
    Function that trains a Siamese model using BCELoss as loss function and Adam as an optimizer 
    For best perormance we use a batch_size = 1 and learning_rate(=lr) = 0.001
    transforms train_classes and train_target to one hot encoding before computing loss
    parameters : aux_loss : boolean to determine if an auxiliary loss should be included
                 w_sharing : boolean to decide if both channel would share same path in the model 
                 alpha : parameter will be used in the linear function of different losses based on the boolean aux_loss 
    
    Returns losses : tensor that contains the train loss per epoch
            test_accuracy :  tensor that contains test accuracy in % per epoch
            train_accuracy : tensor that contains train accuracy in % per epoch
    '''
    criterion = nn.BCELoss() #Better loss function than CrossEntropy
    optimizer = torch.optim.Adam(model.parameters(), lr = lr) # Adam is way better than SGD                                  
    if not aux_loss : alpha = 1 #if alpha = 1 the auxilary loss will not be included 
    losses = torch.zeros(nb_epochs) #contains the loss per epoch
    test_accuracy = torch.zeros(nb_epochs) 
    train_accuracy = torch.zeros(nb_epochs)
    print("training starting now :")
    for e in range(nb_epochs):
        cumul_loss = 0
        #print("epoch " +str(e+1) + "/" + str(nb_epochs) + " in progress..")
        for b , (input , target) in enumerate(zip(train_input.split(batch_size), train_target.split(batch_size))):
            image1, image2 , output = model(input , w_sharing )
            #digit1 : True digit on the first channel ,  digit2 : True digit on the second channel 
            digit1 = train_classes.narrow(0,b*batch_size , batch_size)[:,0]
            digit2 = train_classes.narrow(0,b*batch_size , batch_size)[:,1]
     
            #Converting digit1 , digit2 and target to one hot encoding to use BCELoss()
            digit1_one_hot = convert_to_one_hot_labels(input , train_classes , digit1)
            digit2_one_hot = convert_to_one_hot_labels(input , train_classes , digit2)
            target_one_hot = convert_to_one_hot_labels(input , train_target , target)
            
            #Computing the loss as a linear function of loss_aux1 , loss_aux2 and loss_target using parameter alpha
            loss_aux1 = criterion(image1 ,digit1_one_hot)
            loss_aux2 = criterion(image2 ,digit2_one_hot)
            loss_target = criterion(output , target_one_hot)
            loss = alpha*loss_target + ((1-alpha)/2)*(loss_aux1 + loss_aux2)
            cumul_loss += loss
            
            #update model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        losses[e] = cumul_loss
        test_accuracy[e] = (1- compute_nb_errors_siamese(model, test_input, test_target ) / test_input.size(0) )* 100
        train_accuracy[e] = (1- compute_nb_errors_siamese(model, train_input, train_target ) / train_input.size(0) )* 100
    return  losses , test_accuracy , train_accuracy 

def compute_nb_errors_siamese(model, data_input, data_target  ,batch_size = 100 ):
    '''
    Function that returns number of misclassified inputs of the actual input parameters.
    To be called in each epoch to see the evolution of the Siamese model's accuracy
    '''
    nb_errors = 0
    for b in range(0,data_input.size(0) , batch_size):
        _ , _ , output = model(data_input.narrow(0,b,batch_size))
        _ , pred = output.max(1) 
        for k in range(batch_size):
            if data_target[b + k] != pred[k] : nb_errors += 1
    return nb_errors

    
def test_siamese_model(nb_rounds = 10 , nb_epochs = 25 , w_sharing = True , aux_loss = True , alpha = 0.2):
    '''
    This function will run 10 rounds of random data and weight initialization of the Siamese model
    In each round it will train the randomly generated model's parameter with the random generadet data_input
    Returns : losses         :tensor that contains the train loss per epoch per round
              test_accuracy  :tensor that contains test accuracy in % per epoch per round
              train_accuracy :tensor that contains train accuracy in % per epoch per round
    These tensors will be later used in the visualisation part           
    '''
    losses = torch.zeros(nb_rounds,nb_epochs)
    train_accuracy = torch.zeros(nb_rounds,nb_epochs)
    test_accuracy = torch.zeros(nb_rounds,nb_epochs)
    for round in range(nb_rounds):
        print("generating new model weights and data randomly")
        train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)
        model = Siamese()
        print("Round " + str(round+1) + "/" + str(nb_rounds) + " in progress")
        loss , test_acc, train_acc =  train_siamese_model(model, train_input, train_target ,train_classes, test_input, test_target ,\
                        nb_epochs = nb_epochs ,  w_sharing = w_sharing , aux_loss = aux_loss,  alpha = alpha  )
        losses[round] = loss
        test_accuracy[round] = test_acc
        train_accuracy[round] = train_acc
        print("end round " + str(round+1))
        
    return losses , test_accuracy , train_accuracy 
   
    


def Siamese_visualisation(models_accuracy , models_loss , nb_epochs = 20):
    '''
    Parameters :
    model_accuracy (dictionnary): contains the different 4 settings of siamese model and each has its test accuracy tensor over 10 rounds
    model_loss     (dictionnary): contains the different 4 settings of siamese model and each has its train loss tensor over 10 rounds
    Plot the comparison of the four different seetings of the Siamese model 
    '''
    ep = torch.arange(nb_epochs).numpy()
    plt.figure(figsize=(35, 12))
    plt.subplot(1, 2, 1)
    for key,value in models_accuracy.items():
        acc_mean = value.mean(0).detach().numpy()
        acc_std = value.std(0).detach().numpy()
        plt.plot(ep, acc_mean , label = key)
        plt.legend(fontsize=25)
        plt.fill_between(ep, acc_mean - acc_std, acc_mean + acc_std, alpha = 0.3)
      
    plt.title('Comparaison of models test accuracy per epoch' , fontsize=35 )
    plt.xlabel('Epoch' , fontsize=30)
    plt.ylabel('Accuracy' , fontsize=30)
    plt.grid(axis='y')
    plt.tick_params(axis='both',  labelsize=25)

    plt.subplot(1, 2, 2)
    for key,value in models_loss.items():
        err_std = value.std(0).detach().numpy()
        err_mean = value.mean(0).detach().numpy()
        plt.plot(ep, err_mean , label = key)
        plt.legend(fontsize=25)
        plt.fill_between(ep, err_mean - err_std, err_mean + err_std, alpha = 0.3)
      
    plt.title('Comparaison of models train loss per epoch' , fontsize=35 )
    plt.xlabel('Epoch', fontsize=30)
    plt.ylabel('Loss' , fontsize=30)
    plt.grid(axis='y')
    plt.tick_params(axis='both',  labelsize=25)
    
    plt.show()
    
def hyperparameter_visualisation(accuracy , errors , nb_epochs = 25):
    '''
    These plots helps us determine the best hyperparameters among a list of our choice.
    '''
    epoch = torch.arange(nb_epochs) 
    ep = epoch.numpy()
    err_std = errors.std(0).detach().numpy()
    err_mean = errors.mean(0).detach().numpy()
    acc_std = accuracy.std(0).detach().numpy()
    acc_mean = accuracy.mean(0).detach().numpy()
    plt.figure(figsize=(30, 10))
    
    plt.subplot(1, 2, 1)
    plt.plot(epoch, acc_mean)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(axis='y')
    plt.fill_between(ep, acc_mean - acc_std, acc_mean + acc_std, alpha = 0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epoch, err_mean)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(axis='y')
    plt.fill_between(ep, err_mean - err_std, err_mean + err_std, alpha = 0.3)
    plt.show()