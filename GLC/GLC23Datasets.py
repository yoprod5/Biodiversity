# Author: Benjamin Deneu <benjamin.deneu@inria.fr>
#         Theo Larcher <theo.larcher@inria.fr>
#
# License: GPLv3
#
# Python version: 3.10.6

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from GLC23PatchesProviders import MetaPatchProvider

class PatchesDataset(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
        is_val=False,
        val_size=None,
        nb_labels=None,
        seed=42
    ):
        self.occurences = Path(occurrences)
        self.base_providers = providers
        self.transform = transform
        self.target_transform = target_transform
        self.provider = MetaPatchProvider(self.base_providers, self.transform)

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        if nb_labels:
            self.nb_labels = nb_labels
        else:
            self.nb_labels = np.max(df[label_name].values)+1

        if val_size:
            df_val = df.sample(frac=val_size, random_state=seed)
            df_train = df.drop(df_val.index)

            if is_val:
                df = df_val
            else:
                df = df_train


        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patch = self.provider[item]

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), target
    
    def plot_patch(self, index):
        item = self.items.iloc[index].to_dict()
        self.provider.plot_patch(item)


class PatchesDatasetMultiLabel(PatchesDataset):
    def __init__(self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
        group_columns=('patchID',),
        nb_labels=None
    ):
        super().__init__(occurrences, providers, transform, target_transform, id_name, label_name, item_columns + list(set(group_columns) - set(item_columns)), nb_labels=nb_labels)

        self.group_columns = group_columns
        #self.unique_group = np.unique(self.items[group_column].values)

        # Create an empty dictionary to store the results
        self.plots = {}

        # Iterate through the rows of the DataFrame
        for idx, row in self.items.iterrows():
            # Extract the patchID and speciesId from the row
            patch_id = tuple([row[col] for col in group_columns])
            species_id = self.targets[idx]
            
            # If the patchID is not already a key in the dictionary, add it with an empty list as the value
            if patch_id not in self.plots:
                self.plots[patch_id] = (idx, [])
            
            # Append the speciesId to the list associated with the patchID key
            self.plots[patch_id][1].append(species_id)
        print(self.plots[patch_id])
    
    def __len__(self):
        return len(self.plots)
    
    def __getitem__(self, index):
        idx, species = self.plots[list(self.plots.keys())[index]]
        targets = np.zeros(self.nb_labels)
        targets[species] = 1.0

        item = self.items.iloc[idx].to_dict()
        patch = self.provider[item]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patch).float(), targets

class PatchesDatasetOld(Dataset):
    def __init__(
        self,
        occurrences,
        providers,
        transform=None,
        target_transform=None,
        id_name="glcID",
        label_name="speciesId",
        item_columns=['lat', 'lon', 'patchID'],
    ):
        self.occurences = Path(occurrences)
        self.providers = providers
        self.transform = transform
        self.target_transform = target_transform

        df = pd.read_csv(self.occurences, sep=";", header='infer', low_memory=False)

        self.observation_ids = df[id_name].values
        self.items = df[item_columns]
        self.targets = df[label_name].values

    def __len__(self):
        return len(self.observation_ids)

    def __getitem__(self, index):
        item = self.items.iloc[index].to_dict()

        patches = []
        for provider in self.providers:
            patches.append(provider[item])

        # Concatenate all patches into a single tensor
        if len(patches) == 1:
            patches = patches[0]
        else:
            patches = np.concatenate(patches, axis=0)

        if self.transform:
            patches = self.transform(patches)

        target = self.targets[index]

        if self.target_transform:
            target = self.target_transform(target)

        return torch.from_numpy(patches).float(), target
