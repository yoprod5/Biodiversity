from data.GLC23PatchesProviders import MultipleRasterPatchProvider, RasterPatchProvider, JpegPatchProvider
from data.GLC23Datasets import PatchesDataset


from __future__ import print_function
from __future__ import division
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


data_path = '/home/oyoume/data' # root path of the data
# configure providers
p_hfp_d = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/detailed/', select = ['Built1994_WGS84', 'Lights2009_WGS84']) # take all rasters from human footprint detailed (2 rasters here)
p_bioclim = MultipleRasterPatchProvider(data_path+'EnvironmentalRasters/Climate/BioClimatic_Average_1981-2010/', select=['bio1', 'bio2']) # take only bio1 and bio2 from bioclimatic rasters (2 rasters from the 3 in the folder)
p_hfp_s = RasterPatchProvider(data_path+'EnvironmentalRasters/HumanFootprint/summarized/HFP2009_WGS84.tif') # take the human footprint 2009 summurized raster (a single raster)
p_rgb = JpegPatchProvider(data_path+'SatelliteImages/') # take all sentinel imagery layers (r,g,b,nir = 4 layers)


# create dataset
dataset = PatchesDataset(
    occurrences=data_path+'Presence_only_occurrences/Presences_only_train_sample.csv',
    providers=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

test_dataset = PatchesDataset(
    occurrences=data_path+'Presence_Absences_occurrences/Presences_Absences_train_sample.csv',
    providers=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))


# Initialize the dataloaders for training.

def data_loader(data_path,
                batch_size,
                provider_list=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb),
                random_seed=42,
                valid_size=0.1,
                shuffle=True,
                test=False
               ):
    
     # root path of the data
    # configure providers
  
   # normalize = transforms.Normalize(
   #     mean=[0.4914, 0.4822, 0.4465],
   #     std=[0.2023, 0.1994, 0.2010],
   # )

    # define transforms
    # transform = transforms.Compose([
    #        transforms.Resize((224,224)),
    #        transforms.ToTensor(),
    #        normalize,
    #])

    if test:
        dataset = PatchesDataset(
            occurrences=data_path+'Presence_Absences_occurrences/Presences_Absences_train_sample.csv',
            providers= provider_list)
            
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

        return data_loader
    # create dataset
    dataset = PatchesDataset(
    occurrences=data_path+'Presence_only_occurrences/Presences_only_train_sample.csv',
    providers= provider_list)

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
train_loader, valid_loader = data_loader(data_path='data/sample_data/',
                                         batch_size=128,provider_list=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

test_loader = data_loader(data_path='data/sample_data/',
                              batch_size=128,
                              test=True,provider_list=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

first_batch_tensor, first_batch_labels = next(iter(train_loader))
print(first_batch_labels)
print(first_batch_tensor)
print(first_batch_tensor.shape)


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
    

device = torch.device('cuda')
# Save path for checkpoints
save_path = 'chekpoints/'
# Save path for logs
logdir = 'logs/'

N_Classes = 1
batch_size = 128
learning_rate = 0.001
# Initialize the training parameters.
num_workers = 8 # Number of CPU processes for data preprocessing
lr = 1e-4 # Learning rate
save_freq = 1 # Save checkpoint frequency (epochs)
test_freq = 10 # Test model frequency (iterations)
max_epoch_number = 100 # Number of epochs for training 
# Initialize the model
model = Resnet18(N_Classes)
print(model)
# Switch model to the training mode and move it to GPU.
model.train()
model = model.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.0001, momentum = 0.9) 

# If more than one GPU is available we can use both to speed up the training.
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

os.makedirs(save_path, exist_ok=True)

# Loss function
criterion = nn.BCELoss()
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
    
    
def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            }

epoch = 0
iteration = 0
since = time.time()
print("Début de l'Entrainement {} ".format(since))
while True:
    batch_losses = []
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        targets = targets.unsqueeze(1)
        targets=targets.float()
        optimizer.zero_grad()
        model_result = model(imgs)
        loss = criterion(model_result,targets)
       # print(np.array(model_result))
        #torch.tensor([target]).float() 
        batch_loss_value = loss.item()
        loss.backward()
        optimizer.step()

        batch_losses.append(batch_loss_value)
 
        if iteration % test_freq == 0:
            model.eval()
            with torch.no_grad():
                model_result = []
                targets = []
                for imgs, batch_targets in valid_loader:
                    imgs = imgs.to(device)
                    model_batch_result = model(imgs)
                    model_result.extend(model_batch_result.cpu().numpy())
                    targets.extend(batch_targets.cpu().numpy())
                    print(targets)

            result = calculate_metrics(np.array(model_result), np.array(targets))
            print("epoch:{:2d} iter:{:3d} test: "
                  "micro f1: {:.3f} "
                  "macro f1: {:.3f} ".format(epoch, iteration,
                                              result['micro/f1'],
                                              result['macro/f1'],
                                             ))
            model.train()
        iteration += 1
 
    loss_value = np.mean(batch_losses)
    print("epoch:{:2d} iter:{:3d} train: loss:{:.3f}".format(epoch, iteration, loss_value))
    if epoch % save_freq == 0:
        checkpoint_save(model, save_path, epoch)
    epoch += 1
    if max_epoch_number < epoch:
        break
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Val loss pourcent: {}%'.format())

print('Finished Training')
