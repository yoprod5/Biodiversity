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
from transformers import AutoImageProcessor, Dinov2ForImageClassification
import torch

import os
import copy


data_path = '/Users/yoprod/Desktop/MesRecherches/GeoClef/' # root path of the data
#data_path2 = 'data/zenith/share/GLC/patches/
# root path of the data
# configure providers
p_rgb = JpegPatchProvider(data_path+'SatelliteImages/', select=['red', 'green', 'blue']) # take all sentinel imagery layers (r,g,b,nir = 4 layers)

# create dataset
#dataset = PatchesDataset(
#    occurrences=data_path+'Presence_only_occurrences/Presences_only_train_sample.csv',
#    providers=(p_hfp_d, p_bioclim, p_hfp_s, p_rgb))

dataset = PatchesDatasetMultiLabel(
    occurrences=data_path+'GLC/PA_Labels.csv',
    providers=(p_rgb))


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





#dataset = load_dataset("huggingface/cats-image")
#image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")
model = Dinov2ForImageClassification.from_pretrained("facebook/dinov2-small-imagenet1k-1-layer")

for image,label in enumerate(train_loader)
    inputs = image_processor(image, return_tensors="pt")

   # with torch.no_grad():
    #    logits = model(**inputs).logits
    #    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    list(last_hidden_states.shape)

# model predicts one of the 1000 ImageNet classes
#predicted_label = logits.argmax(-1).item()
#print(model.config.id2label[predicted_label])
#tabby, tabby cat

print('TRAINING COMPLETE')

