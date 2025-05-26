import os
import json
import itertools
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class WaterBirds(Dataset): 
    def __init__(self, root, split="train", transform=None):
        try:
            split_i = ["train", "valid", "test"].index(split)
        except ValueError:
            raise(f"Unknown split {split}")
        self.split = split
        metadata_df = pd.read_csv(os.path.join(root, "metadata.csv"))
        self.metadata_df = metadata_df[metadata_df["split"] == split_i]
        self.root = root
        self.transform = transform
        self.y_array = self.metadata_df['y'].values
        self.classnames = ['landbird', 'waterbird']
        self.p_array = self.metadata_df['place'].values
        self.n_classes = np.unique(self.y_array).size
        self.confounder_array = self.metadata_df['place'].values
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.indicator = np.abs(self.y_array  -  self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        self.filename_array = self.metadata_df['img_filename'].values

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        attr = torch.LongTensor(
            [y, p, g])

        img_path = os.path.join(self.root, self.filename_array[idx])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img, attr, self.filename_array[idx]