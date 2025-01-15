from data.GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from data.GLC23Datasets import PatchesDataset,PatchesDatasetMultiLabel

import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import os
import copy
print('Importation terminée...')

class Resnet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.resnet = models.resnet18(pretrained=True)
        #new_classifier = nn.Sequential(*list(self.resnet.classifier.children())[:-1])
        #self.resnet.classifier = new_classifier
            
        # get the pre-trained weights of the first layer
        pretrained_weights = self.resnet.conv1.weight
        #new_features = nn.Sequential(*list(self.resnet.features.children()))
        #new_features[0] 
        self.resnet.conv1= nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # For M-channel weight should randomly initialized with Gaussian
        #self.resnet.conv1.weight.data.normal_(0, 0.001)
        # For RGB it should be copied from pretrained weights
        #self.resnet.conv1.weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = self.resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    
class Resnet50(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        self.resnet = models.resnet50(pretrained=True)
        #new_classifier = nn.Sequential(*list(self.resnet.classifier.children())[:-1])
        #self.resnet.classifier = new_classifier
            
        # get the pre-trained weights of the first layer
        pretrained_weights = self.resnet.conv1.weight
        #new_features = nn.Sequential(*list(self.resnet.features.children()))
        #new_features[0] 
        self.resnet.conv1= nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 2),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        # For M-channel weight should randomly initialized with Gaussian
        #self.resnet.conv1.weight.data.normal_(0, 0.001)
        # For RGB it should be copied from pretrained weights
        #self.resnet.conv1.weight.data[:, :3, :, :] = nn.Parameter(pretrained_weights)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=self.resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = self.resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    
    
class CNN_TimeSeries(nn.Module):
        def __init__(self,N_Classes):
            super(CNN_TimeSeries,self).__init__(),
            #self.TimeSeries = \n",
            #self.positional_encoding = PositionalEncoding(input_size)
            self.conv1d_1 =  nn.Sequential(
                            nn.Conv1d(6, 32, kernel_size = 2),
                            nn.BatchNorm1d(32))
            self.conv1d_2=  nn.Sequential(
                            nn.Conv1d(32, 64, kernel_size = 2),
                            nn.BatchNorm1d(64))
            self.conv1d_3=  nn.Sequential(
                            nn.Conv1d(64, 128, kernel_size = 2),
                            nn.BatchNorm1d(128))
            #self.batchnorm = nn.BatchNorm1d(64)
            #nn.Conv2d(1,6,kernel_size=2)\n",
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool1d(kernel_size = 2)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc1 = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features= 2432, out_features=128)
            )
            self.fc2 = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=N_Classes))
            self.sigm =nn.Sigmoid()
   
        def forward(self,x):
            #x = self.positional_encoding(x)
            x = self.conv1d_1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.relu1(x)
            x = self.conv1d_2(x)
            x = self.relu(x)
            x = self.conv1d_3(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = x.reshape(x.size(0), -1)
            #x = x.view(-1)\n",
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigm(x)
            #x = self.logSoftmax(x)\n",
            return x


