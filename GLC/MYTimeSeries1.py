from data.GLC23TimeSeriesProviders import MultipleCSVTimeSeriesProvider, CSVTimeSeriesProvider
from data.GLC23Datasets import TimeSeriesDataset

import random
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from data.GLC23TimeSeriesProviders import MultipleCSVTimeSeriesProvider, CSVTimeSeriesProvider
from data.GLC23Datasets import TimeSeriesDataset, TimeSeriesMultilabel, TimeSeriesDatasetbis
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
from tqdm import tqdm
import os
import copy
print('Importation des librairies terminées....')

data_path = '/Users/yoprod/Desktop/MesRecherches/GeoClef/' # root path of the data
# configure providers
#ts_red = CSVTimeSeriesProvider(data_path+'SatelliteTimeSeries/time_series_red.csv')
#ts_multi = MultipleCSVTimeSeriesProvider(data_path+'SatelliteTimeSeries/', select=['red','green','blue'])
ts_all = MultipleCSVTimeSeriesProvider(data_path+'SatelliteTimeSeries/')
print('Chargement des datasets...')
# create dataset
#dataset = TimeSeriesMultilabel(occurrences=data_path+'GLC/PA_Labels.csv',
#      providers=[ts_all])


dataset = TimeSeriesDatasetbis(occurrences=data_path+'Presence_only_occurrences/Presences_only_train.csv',
                            providers=[ts_all])
print('FIn Chargement de la Dataset PO...')
# print random tensors from dataset
#ids = [random.randint(0, len(dataset)-1) for i in range(5)]
#for id in ids:
#    tensor = dataset[id][0]
#    label = dataset[id][1]
#    print('Tensor type: {}, tensor shape: {}, label: {}'.format(type(tensor), tensor.shape, label))
#    dataset.plot_ts(id)
    
def data_loader(dataset,
                batch_size,
                provider_list=(ts_all),
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

print('Chargement du train loader et valid loader effectué ...')
#test_loader = data_loader(data_path='data/sample_data/',
##                              batch_size=128,
#                              test=True,provider_list=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

#first_batch_tensor, first_batch_labels = next(iter(train_loader))
#print(first_batch_labels)
#print(first_batch_tensor)
#print(first_batch_tensor.shape)

class PositionalEncoding(nn.Module):
    def __init__(self, input_size):
        super(PositionalEncoding, self).__init__()
        self.position_enc = nn.Embedding(input_size, input_size)
    
    def forward(self, x):
        #x = x.reshape(x.size(0), -1)
        print(x.size(0))
        seq_length = x.size(0)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_embeddings = self.position_enc(position_ids)
        return x + position_embeddings

# Exemple de modèle CNN avec Positional Encoding
class CNN_TimeSeries(nn.Module):
        def __init__(self,input_size, output_size):
            super(CNN_TimeSeries,self).__init__(),
            #self.TimeSeries = \n",
            self.positional_encoding = PositionalEncoding(input_size)

            self.conv1d_1 =  nn.Sequential(
                            nn.Conv1d(6, 32, kernel_size = 2),
                            nn.BatchNorm1d(32))
            self.conv1d_2=  nn.Sequential(
                            nn.Conv1d(32, 64, kernel_size = 2),
                            nn.BatchNorm1d(64))
            self.conv1d_3=  nn.Sequential(
                            nn.Conv1d(64, 128, kernel_size = 2),
                            nn.BatchNorm1d(128))
            self.conv1d_4=  nn.Sequential(
                            nn.Conv1d(128, 256, kernel_size = 2),
                            nn.BatchNorm1d(256))
            #self.batchnorm = nn.BatchNorm1d(64)
            #nn.Conv2d(1,6,kernel_size=2)\n",
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool1d(kernel_size = 2)
            self.relu1 = nn.ReLU(inplace=True)
            self.fc1 = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features= 4864 , out_features=128)
            )
            self.fc2 = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=128, out_features=output_size))
            #self.sigm =nn.Sigmoid()
   
        def forward(self,x):
           # x = self.positional_encoding(x)
            x = self.conv1d_1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.relu1(x)
            x = self.conv1d_2(x)
            x = self.relu(x)
            x = self.conv1d_3(x)
            x = self.relu(x)
            x = self.conv1d_4(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.relu(x)
            x = x.reshape(x.size(0), -1)
            #x = x.view(-1)\n",
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            #x = self.sigm(x)
            #x = self.logSoftmax(x)\n",
            return x
# Exemple d'utilisation
input_size = 6  # Nombre de bandes
output_size = 10040 # Nombre de classes de sortie
seq_length = 84*6  # Longueur de la série temporelle
        

#device = torch.device('cuda')
# Save path for checkpoints
save_path = 'chekpoints_S/'
# Save path for logs
logdir = 'logs_S/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_Classes = 2174
batch_size = 128
learning_rate = 0.01
# Initialize the training parameters.
num_workers = 8 # Number of CPU processes for data preprocessing
save_freq = 1 # Save checkpoint frequency (epochs)
test_freq = 10 # Test model frequency (iterations)
max_epoch_number = 10 # Number of epochs for training 
epochs = 50

# Initialize the model
model = CNN_TimeSeries(input_size, output_size).to(device)
print(model)
#checkpoint = torch.load(data_path + 'GLC//outputs/best_model2.pth')
# load model weights state_dict
#model.load_state_dict(checkpoint['model_state_dict'])

# Switch model to the training mode and move it to GPU.
model.train()
model = model.to(device)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.001, momentum = 0.9) 


# If more than one GPU is available we can use both to speed up the training.
if torch.cuda.device_count() > 1:
     model = nn.DataParallel(model)

os.makedirs(save_path, exist_ok=True)
os.makedirs(logdir, exist_ok=True)


# Loss function
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
                }, 'outputs/best_model2.pth')
          
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
                }, 'outputs/final_model2.pth')
    
    
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
    plt.savefig('outputs/accuracy2.png')
    
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
    plt.savefig('outputs/loss2.png')
print('Beginning of training ...')
save_best_model = SaveBestModel()
since =time.time()

def Train():
    running_loss = 0
    train_running_correct = 0
    for idx, (inputs,labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.squeeze(1)
        outputs = model(inputs.float())
        #loss = criterion(outputs.type(torch.float),labels.type(torch.float))
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        #loss = Variable(loss, requires_grad = True)
        _, preds = torch.max(outputs.data, 1)
        #print(preds)
        print(preds)
        print('Versius')
        print(outputs)
        loss.backward()
        optimizer.step()
        #result = calculate_metrics(outputs.cpu(), np.array(preds))
        #for metric in result:
        #    logger.add_scalar('train/' + metric, result[metric])
        #print(result['micro/f1'])
        #print("epoch:{:2d}".format(epoch))
        #optimizer.step()
        running_loss += loss
        train_running_correct += (preds == labels).sum().item()
        print(train_running_correct)
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))
    print('Loss : {} Acc: {}'.format(epoch_acc,epoch_loss))
    return epoch_loss, epoch_acc
    #train_losses.append(train_loss.detach().numpy())
    #print(f'train_loss {train_loss}')

    
def Valid():
    running_loss = 0.0
    valid_running_correct = 0
    model.eval()
    valid_losses = []
    with torch.no_grad():
        model_result = []
        targets = []
        for idx, (inputs, labels) in enumerate(valid_loader):
            inputs = inputs.to(device)
            inputs = inputs.squeeze(1)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs,labels)
            #
            _, preds = torch.max(outputs.data, 1)
            running_loss += loss
       # result = calculate_metrics(outputs.cpu().numpy(), np.array(preds.cpu().numpy()))
       # for metric in result:
       #     logger.add_scalar('test/' + metric, result[metric])
       # print("epoch:{:2d} test: "
       #       "micro f1: {:.3f} "
       #       "macro f1: {:.3f} ".format(epoch, 
       #                                   result['micro/f1'],
       #                                   result['macro/f1']))    
        valid_loss = running_loss/len(valid_loader)
        valid_losses.append(valid_loss.detach().numpy())
        valid_running_correct += (preds == labels).sum().item()    
        # loss and accuracy for the complete epoch
    epoch_loss = 100. * sum(valid_losses) / len(valid_losses)
    epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
    #print(f'valid_loss {valid_loss}')
    return epoch_loss, epoch_acc
        
        

# lists to keep track of losses and accuracies
train_loss, valid_losses = [], []
train_acc, valid_acc = [], []
# start the training
print('Debut de lentrainement....')
for epoch in range(epochs):
    print(f"[INFO]: Epoch {epoch+1} of {epochs}")
    train_epoch_loss, train_epoch_acc = Train()
    valid_epoch_loss, valid_epoch_acc = Valid()
    train_loss.append(train_epoch_loss)
    valid_losses.append(valid_epoch_loss)
    train_acc.append(train_epoch_acc)
    valid_acc.append(valid_epoch_acc)
    print(f"Training loss: {train_epoch_loss:.3f}")
    print(f"Training Accuracy: {train_epoch_acc:.3f}")
    print(f"Validation Accuracy: {validation_epoch_acc:.3f}")
    print(f"Validation loss: {valid_epoch_loss:.3f}")
    # save the best model till now if we have the least loss in the current epoch
    save_best_model(
        valid_epoch_loss, epoch, model, optimizer, criterion
    )
    
# save the trained model weights for a final time
save_model(epochs, model, optimizer, criterion)
# save the loss and accuracy plots
save_plots(train_acc, valid_acc, train_loss, valid_losses)

print('TRAINING COMPLETE')

