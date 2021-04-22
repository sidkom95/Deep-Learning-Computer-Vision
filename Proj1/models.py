import torch
import dlc_practical_prologue as prologue
import torch.nn as nn
import torch.nn.functional as F
import statistics

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.fc1 = nn.Linear(14*14*2, 40)#14*14*2 , 40
        self.fc2 = nn.Linear(40, 20) #40,20
        self.fc3 = nn.Linear(20, 2)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.sigmoid(self.fc3(x)) #better activation function for binary classification
        
        return output

class Siamese(nn.Module):
    # We used activation1 and activation 2 during grid search of hyperparameters 
    def __init__(self, activation1 = nn.Tanh() , activation2 = nn.Sigmoid() , rate = 0.3):
        super().__init__()
        #Convolutional network for the image's 1st channel: used twice if w_sharing is True
        
        self.Conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 3),
                                      activation1, #ReLU and Leakyrelu kills a lot of neurons for negative values
                                      nn.AvgPool2d(kernel_size = 2, stride=2),
                                      nn.Dropout(rate),
                                      nn.Conv2d(32, 64, kernel_size = 3),
                                      activation1,
                                      nn.AvgPool2d(kernel_size = 2, stride=2),
                                      nn.Dropout(rate)
                                      )
        # Fully connected network for the image's 1st channel: used twice if w_sharing is True
        self.Fc1 = nn.Sequential(nn.Linear(256, 128),
                                 activation1,
                                 nn.Linear(128, 10),
                                 activation2 #sigmoid is better than softmax
                                 )

        # Convolutional network for the image's 2nd channel: not used if w_sharing is True
        self.Conv2 = nn.Sequential(nn.Conv2d(1, 32, kernel_size = 3),
                                      activation1, 
                                      nn.AvgPool2d(kernel_size = 2, stride=2),
                                      nn.Dropout(rate),
                                      nn.Conv2d(32, 64, kernel_size = 3),
                                      activation1,
                                      nn.AvgPool2d(kernel_size = 2, stride=2),
                                      nn.Dropout(rate)
                                      )
        # Fully connected network for the image's 2nd channel: not used if w_sharing is True
        self.Fc2 = nn.Sequential(nn.Linear(256, 128),
                                 activation1,
                                 nn.Linear(128, 10),
                                 activation2
                                 )

        # Final layer : combine outputs of the two channels
        self.Final = nn.Sequential(nn.Linear(20, 2), 
                                   activation2)
        
        
        
    def predict_labels(self, img , conv, lin):
        img = conv(img)
        img = img.view(img.size(0), -1)
        img = lin(img)
        return img
      
    
    def forward(self, x, w_sharing = True):
        image1 , image2 = torch.chunk(x, chunks=2, dim=1)
        if w_sharing:
            image1 = self.predict_labels(image1, self.Conv1, self.Fc1)
            image2 = self.predict_labels(image2, self.Conv1, self.Fc1)
        else:
            image1 = self.predict_labels(image1, self.Conv1, self.Fc1)
            image2 = self.predict_labels(image2, self.Conv2, self.Fc2)
        output = torch.cat((image1, image2), dim=1)
        output = self.Final(output)
        return image1, image2 , output

  