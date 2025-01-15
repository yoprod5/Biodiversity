from data.GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from data.GLC23Datasets import PatchesDataset, PatchesDatasetMultiLabel

import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_score, recall_score, f1_score

import os
import copy


data_path = '/Users/yoprod/Desktop/MesRecherches/GeoClef/' # root path of the data
#data_path2 = 'data/zenith/share/GLC/patches/
# root path of the data
# configure providers
p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/', select = ['Built1994_WGS84', 'Lights2009_WGS84']) # take all rasters from human footprint detailed (2 rasters here)
p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/', select=['bio1', 'bio2']) # take only bio1 and bio2 from bioclimatic rasters (2 rasters from the 3 in the folder)
p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') # take the human footprint 2009 summurized raster (a single raster)
p_rgb = JpegPatchProvider(data_path+'SatelliteImages/') # take all sentinel imagery layers (r,g,b,nir = 4 layers)


# create dataset
#dataset = PatchesDataset(
#    occurrences=data_path+'Presence_only_occurrences/Presences_only_train_sample.csv',
#    providers=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

dataset = PatchesDatasetMultiLabel(
    occurrences=data_path+'GLC/PA_Labels.csv',
    providers=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))


# Initialize the dataloaders for training.

def data_loader(dataset,
                batch_size,
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False
               ):
    
     # root path of the data
    # configure providers

    train_dataset = dataset
    valid_dataset = dataset
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler)
 
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler)

    return (train_loader, valid_loader)


# CIFAR10 dataset 
train_loader, valid_loader = data_loader(dataset=dataset,
                                         batch_size=128)


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
                        nn.Conv2d(9, 64, kernel_size = 3, stride = 1, padding = 2),
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
    

#device = torch.device('cuda')
# Save path for checkpoints
save_path = 'chekpoints/'
# Save path for logs
logdir = 'logs/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_Classes = 2174
batch_size = 128
learning_rate = 0.01
# Initialize the training parameters.
#num_workers = 8 # Number of CPU processes for data preprocessing
save_freq = 1 # Save checkpoint frequency (epochs)
test_freq = 10 # Test model frequency (iterations)
max_epoch_number = 10 # Number of epochs for training 
epochs=50
thresold = 0.5 


# Initialize the model
model = Resnet18(N_Classes).to(device)
print(model)
# Switch model to the training mode and move it to GPU.
model.train()
#model = model.to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9) 


# If more than one GPU is available we can use both to speed up the training.
if torch.cuda.device_count() > 1:
     model = nn.DataParallel(model)

os.makedirs(save_path, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


# Loss function
#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()


#optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay = 0.001, momentum = 0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#criterion = nn.MSELoss()
# Tensoboard logger
logger = SummaryWriter(logdir)

# Here is an auxiliary function for checkpoint saving.
def checkpoint_save(model, save_path, epoch):
    f = os.path.join(save_path, 'checkpoint-{:06d}.pth'.format(epoch))
    if 'module' in dir(model):
        torch.save(model.module.state_dict(), f)
    else:
        torch.save(model.state_dict(), f)
    print('saved checkpoint:', f)

#Function for saving model

#Function for saving model
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            }

#Saving the best model of training
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/best_model.pth')
          
#Saving the final model

def save_model(epochs, model, optimizer, criterion):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs/final_model.pth')
    
    
    #Save In figure result
def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('outputs/accuracy.png')
    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')
print('Beginning of training ...')
save_best_model = SaveBestModel()
since =time.time()

save_best_model = SaveBestModel()
def Train():
    running_loss = 0
    train_running_correct = 0
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.squeeze(1)
        outputs = model(inputs.float())
        loss = criterion(outputs.type(torch.float),labels.type(torch.float))
        optimizer.zero_grad()
        #loss = Variable(loss, requires_grad = True)
        #_, preds = torch.max(outputs.data, 1)
        #print(preds)
        print(labels)
        print('Versius')
        print(outputs)
        loss.backward()
        optimizer.step()
        result = calculate_metrics(outputs.cpu(), np.array(labels))
        for metric in result:
            logger.add_scalar('train/' + metric, result[metric])
        print(result['micro/f1'])
        print("epoch:{:2d}".format(epoch))
        
        #optimizer.step()
        running_loss += loss
   #     train_running_correct += (preds == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    #epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    return epoch_loss
    #train_losses.append(train_loss.detach().numpy())
    #print(f'train_loss {train_loss}')

    
def Valid():
    running_loss = 0.0
    valid_running_correct = 0
    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            inputs = inputs.squeeze(1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs.type(torch.float),labels.type(torch.float))
            #_, preds = torch.max(outputs.data, 1)
            running_loss += loss
        result = calculate_metrics(outputs.cpu().numpy(), np.array(labels.cpu().numpy()))
        for metric in result:
            logger.add_scalar('test/' + metric, result[metric])
        print("epoch:{:2d} test: "
              "micro f1: {:.3f} "
              "macro f1: {:.3f} ".format(epoch, 
                                          result['micro/f1'],
                                          result['macro/f1']))    
        valid_loss = running_loss/len(valid_loader)
       # valid_losses.append(valid_loss.detach().numpy())
        #valid_running_correct += (preds == labels).sum().item()    
        # loss and accuracy for the complete epoch
    epoch_loss = valid_loss
    #epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    #print(f'valid_loss {valid_loss}')
    return epoch_loss
        

# lists to keep track of losses and accuracies
train_loss, valid_loss = [], []
train_acc, valid_acc = [], []
# start the training
print('Debut de lentrainement....')
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss = Train()
    valid_epoch_loss, valid_epoch_acc = Valid()
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    #train_acc.append(train_epoch_acc)
    #valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}")
    # save the best model till now if we have the least loss in the current epoch
    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion
    )
    
# save the trained model weights for a final time
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
#save_plots(train_acc, valid_acc, train_loss, valid_loss)

print('TRAINING COMPLETE')

